"""Validate and load JSON or YAML settings; paths are relative to project root."""

import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path or os.environ.get("MUNICIPAL_CONFIG", ROOT / "configs/default.json"))
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) if path.suffix in {".yaml", ".yml"} else json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    if value["clustering"]["n_clusters"] < 2 or value["network"]["n_neighbors"] < 1:
        raise ValueError("n_clusters must be >=2; n_neighbors must be >=1")
    if value["threads"] < 1 or value["features"]["share_scale"] <= 0:
        raise ValueError("threads and share_scale must be positive")
    categories = value["features"]["categories"]
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Feature categories must be nonempty and unique")
    return value


CONFIG = load_config()


def project_path(path):
    return ROOT / path
