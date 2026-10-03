"""Property: for any Llama-shaped config and length, the traced matmul FLOPs equal the closed form."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simfront import models
from simfront.capture import trace_dispatch


@st.composite
def llama_configs(draw):
    kv = draw(st.sampled_from([1, 2, 4]))
    heads = kv * draw(st.sampled_from([1, 2, 4]))
    hd = draw(st.sampled_from([8, 16, 32]))
    return models.tiny_llama(num_hidden_layers=draw(st.integers(1, 3)), num_attention_heads=heads,
                             num_key_value_heads=kv, hidden_size=heads * hd,
                             intermediate_size=draw(st.integers(8, 200)), vocab_size=draw(st.integers(16, 3000)))


@pytest.mark.req("SF-03")
@settings(max_examples=30, deadline=None)
@given(llama_configs(), st.integers(1, 96), st.integers(1, 3))
def test_traced_matmul_flops_equal_closed_form(cfg, length, batch):
    tr = trace_dispatch(models.build(cfg), models.tokens(batch, length))[0]
    d, L, f, v = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size
    kv = cfg.num_key_value_heads * (d // cfg.num_attention_heads)
    weights = L * (2 * d * d + 2 * d * kv + 3 * d * f) + v * d
    hd = d // cfg.num_attention_heads
    # rotary frequencies come from position_ids of shape (1, length): computed once, not per sequence
    expected = 2 * batch * length * weights + 4 * L * d * batch * length * length + 2 * (hd // 2) * length
    assert tr.flops_of("matmul") == expected
    assert tr.unknown() == {}
