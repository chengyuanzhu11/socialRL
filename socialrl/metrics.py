from typing import Any, Dict, Iterable

import numpy as np


def summarize_training(logs: Iterable[Dict[str, Any]], threshold: float, window: int) -> Dict[str, float | int | None]:
    """Compute unsmoothed AUC and a rolling-success threshold crossing."""
    rows = list(logs)
    success = np.asarray([row["bout_success_rate"] for row in rows], dtype=float)
    if len(success) == 0:
        return {"episodes_completed": 0, "success_auc": float("nan"), "time_to_threshold": None}
    auc = float(np.trapezoid(success, dx=1.0)) if len(success) > 1 else 0.0
    threshold_episode = None
    for index in range(len(success)):
        start = max(0, index - window + 1)
        if index - start + 1 == window and float(np.mean(success[start:index + 1])) >= threshold:
            threshold_episode = index + 1
            break
    return {
        "episodes_completed": len(rows),
        "success_auc": auc,
        "time_to_threshold": threshold_episode,
        "threshold": float(threshold),
        "threshold_window": int(window),
    }
