from __future__ import annotations


def predict_gicon(processed: dict[str, object]):
    """Mandatory fixed GICON call."""

    return processed["model"].predict(data=processed["data"], graph=processed["graph"])
