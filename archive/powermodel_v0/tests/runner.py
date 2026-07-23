"""Tiny dependency-free test runner.

The cluster numpy-venv has numpy/pandas/matplotlib but no pytest. The test
modules only use ``pytest.approx``, so if real pytest is missing we install a
minimal shim into ``sys.modules`` and then discover/run ``test_*`` functions.

    $SCRATCH/npm-venv/bin/python -m powermodel.tests.runner
"""

from __future__ import annotations

import importlib
import math
import sys
import traceback


def _install_pytest_shim():
    try:
        import pytest  # noqa: F401
        return
    except Exception:
        pass
    import types

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            return math.fabs(other - self.expected) <= max(
                self.rel * math.fabs(self.expected), self.abs)

        def __repr__(self):
            return f"approx({self.expected}, rel={self.rel})"

    shim = types.ModuleType("pytest")
    shim.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)

    class _Mark:
        def parametrize(self, *a, **k):
            def deco(fn):
                return fn
            return deco

    shim.mark = _Mark()
    sys.modules["pytest"] = shim


def main():
    _install_pytest_shim()
    modules = ["powermodel.tests.test_arch", "powermodel.tests.test_workload"]
    passed = failed = 0
    for modname in modules:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {modname}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {modname}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
