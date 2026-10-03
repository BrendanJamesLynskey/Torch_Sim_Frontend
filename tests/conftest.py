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
