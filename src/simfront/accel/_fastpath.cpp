// The tile-pipeline recurrence of simfront.accel.fastpath, in C++20, exposed with pybind11.
//
// It is a line-for-line port of fastpath._recurrence: the same additions and maxima in the
// same order, so its results are bit-identical to the Python recurrence and to the SimPy model
// (tests/test_accel_fastpath.py). The modern-C++ points it uses, explained in the README:
//
//  * templates: MinHeap<T, Key> is a generic binary heap; the free list instantiates it;
//  * move semantics: the columns are moved into the Pipeline and the result columns are
//    moved out to Python, so no vector of a million doubles is copied;
//  * smart pointers: Python owns each Pipeline through std::unique_ptr (pybind11's default
//    holder), so its memory is released when the Python object goes;
//  * C++20: std::span views the columns without copying, and a concept constrains the heap key.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

template <typename K>
concept Ordered = requires(const K& a, const K& b) {
    { a < b } -> std::convertible_to<bool>;
};

// A binary min-heap ordered by key(item). std::priority_queue would do; writing it out keeps
// the pop order identical to Python's heapq for equal keys (both break ties by a sequence number).
template <typename T, Ordered K, K (*key)(const T&)>
class MinHeap {
public:
    void push(T item) {
        items_.push_back(std::move(item));
        std::size_t i = items_.size() - 1;
        while (i > 0) {
            std::size_t parent = (i - 1) / 2;
            if (!(key(items_[i]) < key(items_[parent]))) break;
            std::swap(items_[i], items_[parent]);
            i = parent;
        }
    }
    [[nodiscard]] const T& top() const { return items_.front(); }
    [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
    T pop() {
        T out = std::move(items_.front());
        items_.front() = std::move(items_.back());
        items_.pop_back();
        std::size_t i = 0, n = items_.size();
        for (;;) {
            std::size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < n && key(items_[l]) < key(items_[m])) m = l;
            if (r < n && key(items_[r]) < key(items_[m])) m = r;
            if (m == i) break;
            std::swap(items_[i], items_[m]);
            i = m;
        }
        return out;
    }

private:
    std::vector<T> items_;
};

struct Free {
    double time;
    std::int64_t seq;
    std::int64_t bytes;
};
using FreeKey = std::pair<double, std::int64_t>;
FreeKey free_key(const Free& f) { return {f.time, f.seq}; }

using Column = std::vector<double>;
using Result = std::tuple<Column, Column, Column, Column, Column, Column, Column, Column>;

class Pipeline {
public:
    Pipeline(std::vector<std::int64_t> op, std::vector<bool> first, std::vector<bool> last, std::vector<bool> stores,
             std::vector<std::int64_t> alloc_in, std::vector<std::int64_t> alloc_out, Column load_dur,
             Column comp_dur, Column store_dur, std::vector<std::vector<std::int64_t>> deps, std::int64_t n_ops,
             std::int64_t capacity)
        : op_(std::move(op)), first_(std::move(first)), last_(std::move(last)), stores_(std::move(stores)),
          alloc_in_(std::move(alloc_in)), alloc_out_(std::move(alloc_out)), load_dur_(std::move(load_dur)),
          comp_dur_(std::move(comp_dur)), store_dur_(std::move(store_dur)), deps_(std::move(deps)), n_ops_(n_ops),
          capacity_(capacity) {
        const std::size_t n = op_.size();
        for (std::size_t len : {first_.size(), last_.size(), stores_.size(), alloc_in_.size(), alloc_out_.size(),
                                load_dur_.size(), comp_dur_.size(), store_dur_.size(), deps_.size()})
            if (len != n) throw std::invalid_argument("columns differ in length");
    }

    Result run() const {
        const std::size_t n = op_.size();
        Column issue(n), dep(n), alloc(n), load_end(n), comp_start(n), comp_end(n), store_start(n), store_end(n);
        std::vector<double> op_done(static_cast<std::size_t>(n_ops_), 0.0);
        MinHeap<Free, FreeKey, free_key> frees;
        std::int64_t seq = 0, level = 0;
        double l_end = 0.0, c_end = 0.0, s_end = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double t = l_end;
            issue[i] = t;
            if (first_[i]) {
                for (std::int64_t d : std::span<const std::int64_t>(deps_[i]))
                    if (op_done[static_cast<std::size_t>(d)] > t) t = op_done[static_cast<std::size_t>(d)];
            }
            dep[i] = t;
            const std::int64_t need = alloc_in_[i] + alloc_out_[i];
            while (!frees.empty() && frees.top().time <= t) level -= frees.pop().bytes;
            while (level + need > capacity_) {
                if (frees.empty()) throw std::runtime_error("tile does not fit in the buffer");
                Free f = frees.pop();
                level -= f.bytes;
                if (f.time > t) t = f.time;
            }
            level += need;
            alloc[i] = t;
            if (load_dur_[i] > 0) t = t + load_dur_[i];
            l_end = t;
            load_end[i] = t;
            double c = l_end > c_end ? l_end : c_end;
            comp_start[i] = c;
            if (comp_dur_[i] > 0) c = c + comp_dur_[i];
            c_end = c;
            comp_end[i] = c;
            if (stores_[i]) {
                if (alloc_in_[i]) frees.push({c, seq++, alloc_in_[i]});
                double s = c > s_end ? c : s_end;
                store_start[i] = s;
                if (store_dur_[i] > 0) s = s + store_dur_[i];
                s_end = s;
                store_end[i] = s;
                if (alloc_out_[i]) frees.push({s, seq++, alloc_out_[i]});
                if (last_[i]) op_done[static_cast<std::size_t>(op_[i])] = s;
            } else {
                if (need) frees.push({c, seq++, need});
                store_start[i] = store_end[i] = c;
            }
        }
        return {std::move(issue), std::move(dep), std::move(alloc), std::move(load_end), std::move(comp_start),
                std::move(comp_end), std::move(store_start), std::move(store_end)};
    }

private:
    std::vector<std::int64_t> op_;
    std::vector<bool> first_, last_, stores_;
    std::vector<std::int64_t> alloc_in_, alloc_out_;
    Column load_dur_, comp_dur_, store_dur_;
    std::vector<std::vector<std::int64_t>> deps_;
    std::int64_t n_ops_, capacity_;
};

}  // namespace

PYBIND11_MODULE(_fastpath, m) {
    m.doc() = "C++20 port of simfront.accel.fastpath's tile-pipeline recurrence";
    m.attr("pybind11_version") = std::to_string(PYBIND11_VERSION_MAJOR) + "." +
                                 std::to_string(PYBIND11_VERSION_MINOR) + "." + std::to_string(PYBIND11_VERSION_PATCH);
    m.attr("cplusplus") = static_cast<long>(__cplusplus);
    py::class_<Pipeline, std::unique_ptr<Pipeline>>(m, "Pipeline")
        .def(py::init<std::vector<std::int64_t>, std::vector<bool>, std::vector<bool>, std::vector<bool>,
                      std::vector<std::int64_t>, std::vector<std::int64_t>, Column, Column, Column,
                      std::vector<std::vector<std::int64_t>>, std::int64_t, std::int64_t>())
        .def("run", &Pipeline::run, py::call_guard<py::gil_scoped_release>(),
             "Per-tile (issue, dep, alloc, load_end, comp_start, comp_end, store_start, store_end)");
    m.def(
        "run",
        [](std::vector<std::int64_t> op, std::vector<bool> first, std::vector<bool> last, std::vector<bool> stores,
           std::vector<std::int64_t> alloc_in, std::vector<std::int64_t> alloc_out, Column load_dur, Column comp_dur,
           Column store_dur, std::vector<std::vector<std::int64_t>> deps, std::int64_t n_ops, std::int64_t capacity) {
            auto p = std::make_unique<Pipeline>(std::move(op), std::move(first), std::move(last), std::move(stores),
                                                std::move(alloc_in), std::move(alloc_out), std::move(load_dur),
                                                std::move(comp_dur), std::move(store_dur), std::move(deps), n_ops,
                                                capacity);
            py::gil_scoped_release release;
            return p->run();
        },
        "Build a Pipeline from the program's columns and run it");
}
