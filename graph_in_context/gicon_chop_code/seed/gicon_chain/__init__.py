"""Inference-time operator chain around a frozen GICON checkpoint."""

from __future__ import annotations

from .gicon_predict import predict_gicon
from .postprocess import postprocess_prediction
from .preprocess import preprocess_context

__all__ = ["run_gicon_chain"]


def run_gicon_chain(context: dict[str, object]):
    """Run the editable F -> fixed GICON -> editable G chain."""

    processed = preprocess_context(context)
    prediction = predict_gicon(processed)
    return postprocess_prediction(prediction, processed)
