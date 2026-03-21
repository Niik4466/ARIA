import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for MCP external tools."""
    
    def load(self, path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config from {path}: {e}")
            return {}
            
    def save(self, path: str, config: Dict[str, Any]) -> None:
        """Saves configuration to a JSON file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save MCP config to {path}: {e}")
