"""Inference-time operator chain around a frozen ICON checkpoint.

The fixed structural spine is:

    preprocess_context(context) -> predict_icon(processed) -> postprocess_prediction(...)

Evolution should mutate preprocessing/postprocessing primitives, gates, and
candidate specs while preserving the mandatory main ICON call.
"""

from __future__ import annotations

from .icon_predict import predict_icon
from .postprocess import postprocess_prediction
from .preprocess import preprocess_context

__all__ = ["run_icon_chop"]


def run_icon_chop(context: dict[str, object]):
    """Run the editable F -> fixed ICON -> editable G chain.

    Contract:
    - ``preprocess_context`` must pass ``context["model"]`` through under
      ``processed["model"]``.
    - ``predict_icon`` is the single mandatory main ICON call.
    - ``postprocess_prediction`` must return shape ``(B, N, 1)`` and finite
      values only.
    """

    processed = preprocess_context(context)
    icon_prediction = predict_icon(processed)
    return postprocess_prediction(icon_prediction, processed)
