"""The NTT golden model, the polynomial-product workload and the in-transit stage."""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simfront.accel import lower, preset, simulate
from simfront.accel.ntt import _is_prime, intt, ntt, ntt_prime, polymul, polymul_schoolbook, polymul_trace, psi


@pytest.mark.parametrize("n", [8, 64, 1024, 1 << 16])
def test_prime_and_root(n):
    q = ntt_prime(n)
    assert _is_prime(q) and (q - 1) % (2 * n) == 0 and q.bit_length() >= 30
    p = psi(n, q)
    assert pow(p, 2 * n, q) == 1 and pow(p, n, q) == q - 1          # primitive 2n-th root


@pytest.mark.req("SF-25")
@settings(max_examples=40, deadline=None)
@given(st.sampled_from([4, 8, 16, 32, 64]), st.integers(0, 2**32))
def test_polymul_equals_schoolbook(n, seed):
    q = ntt_prime(n)
    rng = random.Random(seed)
    a = [rng.randrange(q) for _ in range(n)]
    b = [rng.randrange(q) for _ in range(n)]
    assert polymul(a, b, q) == polymul_schoolbook(a, b, q)


def test_ntt_roundtrip():
    n = 256
    q = ntt_prime(n)
    w = pow(psi(n, q), 2, q)
    a = [random.Random(3).randrange(q) for _ in range(n)]
    assert intt(ntt(a, q, w), q, w) == a


def test_x_to_the_n_is_minus_one():
    n = 16
    q = ntt_prime(n)
    x = [0] * n
    x[n - 1] = 1                                          # X^(n-1) * X = X^n = -1
    y = [0] * n
    y[1] = 1
    assert polymul(x, y, q) == [q - 1] + [0] * (n - 1)


def test_polymul_trace_shape():
    tr = polymul_trace(log_n=12, limbs=3, products=2)
    assert len(tr.ops) == 2 * 3 * 4
    ntts = [o for o in tr.ops if o.category == "ntt"]
    assert len(ntts) == 2 * 3 * 3
    assert all(o.attrs["butterflies"] == 2048 * 12 and o.flops == 3 * 2048 * 12 for o in ntts)


@pytest.mark.req("SF-26")
@pytest.mark.parametrize("budget", [0.5, 1.6, 3.0, 12.0])
def test_in_transit_stage_respects_its_budget(budget):
    tr = polymul_trace(log_n=14, limbs=2)
    cfg = preset("edge-npu", transit_ops_per_byte=budget)
    prog = lower(tr, cfg)
    for x in prog.tiles:
        op = prog.ops[x.op]
        if op.category == "ntt":
            assert x.unit == "transit" and x.comp_dur == 0
            ops = 3 * tr.ops[op.index].attrs["butterflies"]
            assert x.load_dur >= ops / (budget * cfg.path_bw)       # never faster than the stage allows
            assert x.load_dur >= x.load_bytes / cfg.path_bw
        else:
            assert x.unit == "vector"
    simulate(prog)


def test_ntt_on_the_vector_unit_without_a_stage():
    prog = lower(polymul_trace(log_n=14, limbs=2), preset("edge-npu"))
    assert {x.unit for x in prog.tiles} == {"vector"}
