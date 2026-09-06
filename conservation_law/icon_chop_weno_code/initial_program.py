"""Entry point for the unified ICON-CHOP solver on the 1D conservation-law benchmark.

Exports both `run_icon_chop` and `run_icon_agent` so the same module is
compatible with the existing evaluator entry hooks.
"""

from icon_chop import run_icon_chop

run_icon_agent = run_icon_chop

__all__ = ["run_icon_chop", "run_icon_agent"]
