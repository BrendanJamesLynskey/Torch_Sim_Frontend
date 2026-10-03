import pytest

from simfront.trace import Trace


@pytest.mark.req("SF-01")
def test_json_round_trip(llama8b_prefill):
    t = Trace.from_json(llama8b_prefill.to_json())
    assert (t.flops, t.bytes, t.weight_bytes, len(t.ops)) == (llama8b_prefill.flops, llama8b_prefill.bytes,
                                                              llama8b_prefill.weight_bytes, len(llama8b_prefill.ops))
    assert t.ops[100] == llama8b_prefill.ops[100]
