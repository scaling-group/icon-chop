"""Entry point used by ICON-CHOP evaluators.

The MFC evaluator imports ``run_icon_chop``; the WENO post-eval imports
``run_icon_agent``.  Keep this file tiny: all mutable operator-chain logic
lives inside ``seed/icon_chop``.
"""

from icon_chop import run_icon_chop

run_icon_agent = run_icon_chop

__all__ = ["run_icon_chop", "run_icon_agent"]
