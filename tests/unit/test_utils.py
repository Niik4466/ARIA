import pytest
import os
import json
from unittest.mock import patch

def test_config_singleton(clean_config):
    from src.utils import Config
    
    # Assert singleton pattern holds
    c1 = Config()
    c2 = Config()
    assert id(c1) == id(c2), "Config should be a singleton"

def test_config_reload_with_mocked_file(clean_config):
    from src.utils import Config, CONFIG_FILE
    
    fake_config_data = {
        "SYSTEM": {
            "USE_RVC": False,
            "verbose_mode": True
        },
        "ASR": {
            "ASR_MODEL_ID": "fake_model"
        }
    }
    
    from unittest.mock import mock_open
    
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_config_data))):
        with patch("os.path.exists", return_value=True):
            # Instantiate config, which will call reload
            config = Config()
            
            # Assert attribute resolution works
            assert config.SYSTEM.USE_RVC is False
            assert config.ASR.ASR_MODEL_ID == "fake_model"
            
            # Assert flat access backward compatibility
            assert config.verbose_mode is True
            assert config.get("USE_RVC") is False

def test_config_update_key(clean_config):
    from src.utils import Config
    
    fake_config_data = {"TEST": {"key1": "val1"}}
    
    from unittest.mock import mock_open
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_config_data))) as mock_file:
        with patch("os.path.exists", return_value=True):
            config = Config()
            
            # Change out the write mock
            mock_write = mock_open()
            with patch("builtins.open", mock_write):
                config.update_key("TEST", "key1", "val2")
                
                assert config.TEST.key1 == "val2"
                assert config.get("key1") == "val2"
