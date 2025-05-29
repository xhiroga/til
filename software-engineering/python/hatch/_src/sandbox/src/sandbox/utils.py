"""Utility functions for the sandbox package."""
import os
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) ->Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_directory(path: Path) ->None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() ->Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent
