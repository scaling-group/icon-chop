"""Fixed spine ICON call.

Do not skip or replace this call.  Auxiliary ICON probes belong in preprocess.py
or postprocess.py; this function is the mandatory main operator action.
"""

from __future__ import annotations


def predict_icon(processed: dict[str, object]):
    """Invoke the frozen ICON wrapper on the processed data dict."""

    return processed["model"].predict(data=processed["data"])
