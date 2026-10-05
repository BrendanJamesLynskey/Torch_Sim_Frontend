"""Build the optional C++ fast path (src/simfront/accel/_fastpath.cpp) with pybind11.

Everything else is configured in pyproject.toml. ``optional=True``: if no C++20 compiler is
available the install still succeeds and simfront.accel falls back to the Python recurrence.
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(ext_modules=[Pybind11Extension("simfront.accel._fastpath", ["src/simfront/accel/_fastpath.cpp"], cxx_std=20,
                                     extra_compile_args=["-O2"], optional=True)])
