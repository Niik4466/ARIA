"""
Utilities Module

Provides helper functions for the ARIA assistant, such as configuration management.
"""

import os
import json

# Define the absolute path to config.json
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

class ConfigNode:
    """Helper class to navigate nested config dictionaries as attributes."""
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return ConfigNode(val)
            return val
        raise AttributeError(f"No configuration attribute '{name}'")

    def get(self, name, default=None):
        return self._data.get(name, default)

    def as_dict(self):
        return self._data

class Config:
    """
    Configuration Manager class.
    Loads and provides easy attribute-style access to the config.json variables, 
    accounting for categories.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self._config = {}
        self._flat_map = {}
        self.reload()

    def reload(self):
        """Reloads the configuration from the JSON file."""
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            self._config = json.load(f)
            
        # Build flat map for backwards compatibility tracking
        self._flat_map = {}
        for category, items in self._config.items():
            if isinstance(items, dict):
                for k, v in items.items():
                    self._flat_map[k] = (category, v)

    def __getattr__(self, name):
        """Allows config.CATEGORY.VARIABLE syntax."""
        if name in self._config:
            val = self._config[name]
            if isinstance(val, dict):
                return ConfigNode(val)
            return val
            
        # Optional: flat access fallback if not found as direct category
        if name in self._flat_map:
            return self._flat_map[name][1]
            
        raise AttributeError(f"'Config' object has no attribute '{name}'")
        
    def get(self, name, default=None):
        """Dictionary-like get method. Checks top level, then flat map."""
        if name in self._config:
            return self._config.get(name)
        if name in self._flat_map:
            return self._flat_map[name][1]
        return default

    def update_key(self, category: str, key: str, value: any):
        """Updates a key within a category and automatically writes to disk."""
        if category not in self._config:
            self._config[category] = {}
        self._config[category][key] = value
        self._flat_map[key] = (category, value)
        
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4)

# Create a global instance
config = Config()
