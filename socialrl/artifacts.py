import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def save_episode_logs(path: str | Path, logs: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in logs]
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(agent: Any, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"metadata": dict(metadata), "agent_type": type(agent).__name__}
    if hasattr(agent, "q"):
        payload.update({
            "model": agent.q.state_dict(),
            "target_model": agent.qt.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "epsilon": agent.eps,
        })
    elif hasattr(agent, "net"):
        payload.update({"model": agent.net.state_dict(), "optimizer": agent.opt.state_dict()})
    elif hasattr(agent, "Q"):
        payload["q_table"] = {key: value.copy() for key, value in agent.Q.items()}
        payload["epsilon"] = agent.eps
    else:
        raise TypeError(f"Unsupported agent type: {type(agent).__name__}")
    return payload


def save_checkpoint(path: str | Path, agent: Any, metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(checkpoint_payload(agent, metadata), path)
