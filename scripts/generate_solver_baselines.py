#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from solver_baseline_harness import write_all_baselines  # noqa: E402


def main() -> None:
    for path in write_all_baselines():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
