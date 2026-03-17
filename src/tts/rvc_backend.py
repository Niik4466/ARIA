import numpy as np
from typing import Optional

from ..utils import Config
config = Config()

RVC_MODEL = config.get("RVC_MODEL")

from src.rvc.inference import RVC

verbose_mode = config.get("verbose_mode")
_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

class RVC_Backend:
    """
    Backend for voice transformation using local RVC Orchestrator.
    """
    def __init__(self, model_name: str = RVC_MODEL):
        self.model_name = model_name
        self.is_available = False
        
        try:
            self.engine = RVC(model_name=self.model_name)
            self.is_available = True
        except Exception as e:
            print(f"[RVC Backend] Initialization Error: {e}")

    def load_model(self, model_name: Optional[str] = None):
        if model_name:
            self.model_name = model_name
            
        if hasattr(self, 'engine'):
            self.engine.load_model(self.model_name)

    def transform_numpy(self, audio_numpy: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if not self.is_available or not hasattr(self, 'engine'):
            return audio_numpy, sr
            
        return self.engine.transform_numpy(audio_numpy, sr)
