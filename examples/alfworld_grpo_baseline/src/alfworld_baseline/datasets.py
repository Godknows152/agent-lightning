"""ALFWorld parquet loading and task-file integrity checks."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

def _row_game_file(row: dict[str, Any]) -> str | None:
    """Read source-task paths from either source parquet or VERL parquet."""
    if row.get("game_file"):
        return str(row["game_file"])
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("game_file"):
        return str(extra["game_file"])
    nested = extra.get("tools_kwargs", {}) if isinstance(extra, dict) else {}
    if isinstance(nested, dict):
        kwargs = nested.get("alfworld_action", {})
        create = kwargs.get("create_kwargs", {}) if isinstance(kwargs, dict) else {}
        if isinstance(create, dict) and create.get("game_file"):
            return str(create["game_file"])
    return None

def load_tasks(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    parquet_path = Path(path).resolve()
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        raise ValueError(f"ALFWorld parquet is empty: {path}")
    rows = frame.to_dict(orient="records")[:limit if limit is not None else None]
    for row in rows:
        raw_game_file = _row_game_file(row)
        if not raw_game_file:
            raise ValueError(f"ALFWorld row lacks game_file: {path}")
        game_file = Path(raw_game_file)
        if not game_file.is_absolute():
            # agl_envs parquet files store paths relative to contrib/recipes/envs.
            game_file = parquet_path.parents[3] / game_file
        if not game_file.is_file():
            raise FileNotFoundError(game_file)
        for required in ("traj_data.json", "game.tw-pddl"):
            if not (game_file.parent / required).is_file():
                raise FileNotFoundError(game_file.parent / required)
    return [dict(row) for row in rows]
