"""Compatibility entry point for the temperature sweep experiment.

The implementation lives in `hnc_lab.experiments.temperature_sweep`.
New code should import from that experiment package.
"""

from hnc_lab.experiments.temperature_sweep.runner import main, run_temperature_sweep

__all__ = [
    "main",
    "run_temperature_sweep",
]


if __name__ == "__main__":
    raise SystemExit(main())
