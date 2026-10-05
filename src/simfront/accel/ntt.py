"""An FHE-style workload: large polynomial products through the number-theoretic transform.

Lattice-based homomorphic encryption (CKKS, BGV, BFV) spends most of its time multiplying
polynomials in Z_q[X]/(X^N + 1) with N = 2^12 to 2^17, each stored in RNS form as one row
of N word-sized residues per prime ("limb"). The fast way is the NTT, the finite-field FFT:

    c = INTT( NTT(a) * NTT(b) )        (negacyclic: twist by psi^i first, psi^2N = 1)

This module has two halves:

* **functional**: an iterative NTT, its inverse and the negacyclic product, checked against
  schoolbook multiplication in ``tests/test_accel_ntt.py``. It is the golden model a
  hardware NTT would be verified against.
* **workload**: :func:`polymul_trace` builds the operator trace of ``products`` RNS polynomial
  products (per limb: NTT a, NTT b, pointwise multiply, INTT), in the same trace format as the
  PyTorch and ONNX front ends, so it runs through the same accelerator model. NTT operators
  carry their butterfly count; the cost convention is three integer operations per butterfly
  (one modular multiply, one modular add, one modular subtract).
"""

from __future__ import annotations

from ..trace import Op, TensorMeta, Trace


# ── functional model ─────────────────────────────────────────────────────
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):        # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ntt_prime(n: int, bits: int = 30) -> int:
    """The smallest prime q = k * 2n + 1 with at least ``bits`` bits (so q = 1 mod 2n)."""
    k = max(1, (1 << (bits - 1)) // (2 * n))
    while not _is_prime(k * 2 * n + 1):
        k += 1
    return k * 2 * n + 1


def _factors(n: int) -> set[int]:
    out, f = set(), 2
    while f * f <= n:
        while n % f == 0:
            out.add(f)
            n //= f
        f += 1
    if n > 1:
        out.add(n)
    return out


def psi(n: int, q: int) -> int:
    """A primitive 2n-th root of unity mod q."""
    order = q - 1
    fs = _factors(order)
    for g in range(2, q):
        if all(pow(g, order // f, q) != 1 for f in fs):
            return pow(g, order // (2 * n), q)
    raise ValueError("no generator")


def _bitrev(a: list[int]) -> list[int]:
    n = len(a)
    bits = n.bit_length() - 1
    return [a[int(f"{i:0{bits}b}"[::-1], 2)] for i in range(n)] if bits else list(a)


def ntt(a: list[int], q: int, w: int) -> list[int]:
    """Iterative radix-2 Cooley-Tukey NTT (cyclic), w a primitive len(a)-th root of unity."""
    n = len(a)
    a = _bitrev([x % q for x in a])
    m = 2
    while m <= n:
        wm = pow(w, n // m, q)
        for k in range(0, n, m):
            x = 1
            for j in range(m // 2):
                u, v = a[k + j], a[k + j + m // 2] * x % q
                a[k + j], a[k + j + m // 2] = (u + v) % q, (u - v) % q
                x = x * wm % q
        m *= 2
    return a


def intt(a: list[int], q: int, w: int) -> list[int]:
    n = len(a)
    inv_n = pow(n, q - 2, q)
    return [x * inv_n % q for x in ntt(a, q, pow(w, q - 2, q))]


def polymul(a: list[int], b: list[int], q: int) -> list[int]:
    """a * b in Z_q[X]/(X^n + 1) through the NTT (negacyclic twist by psi)."""
    n = len(a)
    p = psi(n, q)
    w = p * p % q
    tw = [pow(p, i, q) for i in range(n)]
    fa = ntt([x * t % q for x, t in zip(a, tw, strict=True)], q, w)
    fb = ntt([x * t % q for x, t in zip(b, tw, strict=True)], q, w)
    c = intt([x * y % q for x, y in zip(fa, fb, strict=True)], q, w)
    inv = pow(p, q - 2, q)
    return [x * pow(inv, i, q) % q for i, x in enumerate(c)]


def polymul_schoolbook(a: list[int], b: list[int], q: int) -> list[int]:
    n = len(a)
    c = [0] * n
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            k = i + j
            if k < n:
                c[k] = (c[k] + x * y) % q
            else:
                c[k - n] = (c[k - n] - x * y) % q
    return c


# ── workload ─────────────────────────────────────────────────────────────
OPS_PER_BUTTERFLY = 3


def _ntt_op(name: str, src: TensorMeta, dst: TensorMeta, log_n: int) -> Op:
    n = 1 << log_n
    bfly = (n // 2) * log_n
    return Op(name, [src], [dst], {"butterflies": bfly, "log_n": log_n}, "ntt", OPS_PER_BUTTERFLY * bfly,
              src.nbytes, dst.nbytes, 0)


def polymul_trace(log_n: int = 16, limbs: int = 24, products: int = 1, word: int = 8) -> Trace:
    """The trace of ``products`` RNS polynomial products, ``limbs`` primes each, N = 2^log_n."""
    n = 1 << log_n
    ops: list[Op] = []
    for p in range(products):
        for limb in range(limbs):
            def t(tag, p=p, limb=limb):
                return TensorMeta(f"p{p}.{tag}.{limb}", (n,), word, False)

            ops.append(_ntt_op("fhe.ntt", t("a"), t("A"), log_n))
            ops.append(_ntt_op("fhe.ntt", t("b"), t("B"), log_n))
            a, b, c = t("A"), t("B"), t("C")
            ops.append(Op("fhe.mulmod", [a, b], [c], {}, "elementwise", n, a.nbytes + b.nbytes, c.nbytes, 0))
            ops.append(_ntt_op("fhe.intt", t("C"), t("c"), log_n))
    return Trace(f"polymul N=2^{log_n} x {limbs} limbs x {products}", "fhe", ops,
                 {"log_n": log_n, "limbs": limbs, "products": products})
