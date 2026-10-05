import logging
import warnings

import pytest

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


@pytest.fixture(scope="session")
def llama8b_prefill():
    """Llama-3-8B, 512-token prefill, traced on the meta device (shared: it is used by several tests)."""
    from simfront import models
    from simfront.capture import trace_dispatch

    m = models.build("llama3-8b")
    return trace_dispatch(m, models.tokens(1, 512), model_name="llama3-8b")[0]


def pytest_configure(config):
    config.addinivalue_line("markers", "req(*ids): the requirements in docs/spec.md this test verifies")


@pytest.fixture(scope="session")
def accel_traces():
    """Small traces for the accelerator model: a CNN (convolutions), a tiny Llama and a tiny GPT-2 (export route)."""
    import torch

    from simfront import models
    from simfront.capture import trace_export

    cnn = models.tiny_cnn()
    out = {"cnn": trace_export(cnn, (models.image(),), model_name="tiny-cnn")}
    for name, cfg in (("llama", models.tiny_llama()), ("gpt2", models.tiny_gpt2())):
        m = models.build(cfg, device="cpu", dtype=torch.float32)
        out[name] = trace_export(m, (models.tokens(1, 32, "cpu"),), {"use_cache": False}, model_name=name)
    return out
