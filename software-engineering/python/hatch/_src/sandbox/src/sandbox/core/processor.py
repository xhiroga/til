"""Core processing functionality."""

from ..utils import load_config, get_project_root
from .. import utils


class DataProcessor:
    """Process data using various utilities."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = load_config(config_file)
        self.root = get_project_root()
    
    def process(self, data: dict) -> dict:
        """Process input data and return processed result."""
        return {
            'name': data.get('name', 'default'),
            'value': data.get('value', 0),
            'root_path': str(self.root)
        }
