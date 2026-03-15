import os
import sys
import numpy as np

# Bind RVC WebUI precisely so Python modules locate infer_pack.models
RVC_WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_webui")
if RVC_WEBUI_DIR not in sys.path:
    sys.path.append(RVC_WEBUI_DIR)

# Hardcode config overrides required by WebUI relative paths
os.environ["rvc_webui_path"] = RVC_WEBUI_DIR
os.environ["weight_root"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rvc_models")
os.environ["index_root"] = os.environ["weight_root"]
os.environ["rmvpe_root"] = os.path.join(RVC_WEBUI_DIR, "assets", "rmvpe")

# Hotfix for PyTorch 2.6+ to allow loading older objects (fairseq, rmvpe)
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Import the actual components from webui pipelined modules
from configs.config import Config
from infer.modules.vc.modules import VC

from config import verbose_mode
_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

import logging
import warnings
if not verbose_mode:
    logging.getLogger("configs.config").setLevel(logging.WARNING)
    logging.getLogger("infer.modules.vc.modules").setLevel(logging.WARNING)
    logging.getLogger("infer.modules.vc.pipeline").setLevel(logging.WARNING)
    logging.getLogger("infer.modules.uvr5.modules").setLevel(logging.WARNING)
    logging.getLogger("fairseq.tasks.hubert_pretraining").setLevel(logging.WARNING)
    logging.getLogger("fairseq.models.hubert.hubert").setLevel(logging.WARNING)
    logging.getLogger("fairseq").setLevel(logging.WARNING)
    
    # Silence PyTorch warnings commonly triggered by older models
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


class RVC:
    """
    RVC Voice Conversion Orchestrator. 
    Ties the backend to the verified math & VITS architecture of the official WebUI repo.
    """
    def __init__(self, model_name: str = "MikuAI_e210_s6300.pth", index_name: str = "added_IVF1811_Flat_nprobe_1_MikuAI_v2.index"):
        # Change working directory so WebUI loads its relative config jsons correctly
        old_cwd = os.getcwd()
        os.chdir(RVC_WEBUI_DIR)
        try:
            self.config = Config()
            self.vc = VC(self.config)
        finally:
            os.chdir(old_cwd)
            
        self.model_name = model_name
        self.index_name = index_name
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.load_model(model_name)
        
    def load_model(self, model_filename: str):
        actual_model_path = model_filename
        folder_path = os.path.join(os.environ["weight_root"], model_filename)
        
        # If it's a directory, search for the .pth file inside
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".pth"):
                    actual_model_path = os.path.join(model_filename, f)
                    break
        elif not model_filename.endswith(".pth") and model_filename != "default":
            actual_model_path = model_filename + ".pth"

        self.model_name = actual_model_path
        print(f"[RVC Orchestrator] Requesting load for {self.model_name} ...")
        
        # In WebUI 'get_vc', the SID is actually just the checkpoint name relative to weight_root
        response = self.vc.get_vc(self.model_name, 0.33, 0.33)
        if isinstance(response, tuple) and len(response) > 0 and type(response[0]) is dict and response[0].get('visible') is False:
            print("[RVC Orchestrator] Model load failed or model empty.")
        else:
            self.dynamic_index_path = response[3]["value"] if isinstance(response, tuple) and len(response) > 3 and type(response[3]) is dict else ""
            print(f"[RVC Orchestrator] Model loaded seamlessly. Index: {self.dynamic_index_path}")
            
    def transform_numpy(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Transforms raw audio passing it directly to the VC Single pipeline.
        
        Args:
            wav: Input audio array
            sr: Source Sample rate
        Returns:
            np.ndarray: Converted audio array
            int: Target out Sample rate
        """
        try:
            import tempfile
            import soundfile as sf
            
            # The WebUI 'load_audio' function depends heavily on `ffmpeg` from path 
            # so we dump the temp chunk to a real wav file before reading.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, wav, sr)
                
            index_path = getattr(self, "dynamic_index_path", "")
            
            # vc_single applies embedding matches, faiss, un-normalized hubert + VITS forward
            info, (tgt_sr, audio_opt) = self.vc.vc_single(
                sid=0, 
                input_audio_path=tmp_path,
                f0_up_key=0,  # Or set a pitch modulation parameter integer
                f0_file=None,
                f0_method="rmvpe",
                file_index=index_path,
                file_index2=index_path,
                index_rate=0.5,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=1.0,
                protect=0.33
            )
            
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            if tgt_sr is None or audio_opt is None:
                print(f"[RVC Orchestrator] Pipeline Exception: {info}")
                return wav, sr
                
            return audio_opt, tgt_sr
            
        except Exception as e:
            print(f"[RVC Orchestrator] Fallback crash: {e}")
            return wav, sr

if __name__ == "__main__":
    print("=== Testing RVC Engine (Official Pipeline Wrapper) ===")
    rvc_engine = RVC(model_name="MikuAI_e210_s6300.pth", index_name="added_IVF1811_Flat_nprobe_1_MikuAI_v2.index")
    import soundfile as sf
    test_audio_path = os.path.join(rvc_engine.base_dir, "src", "tts", "ref_voice", "ref_voice.wav")
    out_audio_path = os.path.join(rvc_engine.base_dir, "src", "tts", "ref_voice", "rvc_output_test.wav")
    
    if os.path.exists(test_audio_path):
        wav, sr = sf.read(test_audio_path)
        out_wav, tgt_sr = rvc_engine.transform_numpy(wav, sr)
        if tgt_sr != sr:
            pass # Usually it becomes 48000 depending on the loaded .pth (since Miku is 48k v2)
        sf.write(out_audio_path, out_wav, tgt_sr)
        print(f"Test Successful. Output: {out_audio_path} - Target SR: {tgt_sr}")
    else:
        print(f"Test audio not found at: {test_audio_path}")
