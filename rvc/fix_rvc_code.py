import os
import sys
import re
import site

def fix_rvc_utils():
    print("[FIX] Searching for rvc_python installation...")
    
    # Try to find the site-packages directory where rvc_python is installed
    # We use site.getsitepackages() or checking import path
    try:
        import rvc_python
        package_path = os.path.dirname(rvc_python.__file__)
        print(f"[FIX] Found rvc_python at: {package_path}")
    except ImportError:
        print("[FIX] rvc_python not found. Please install it first (pip install rvc-python).")
        return

    # Target file: rvc_python/modules/vc/utils.py
    target_file = os.path.join(package_path, "modules", "vc", "utils.py")
    
    if not os.path.exists(target_file):
        print(f"[FIX] Target file not found: {target_file}")
        return

    print(f"[FIX] Patching file: {target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The new function code
    new_function = """
def load_hubert(config,lib_dir):
    import torch
    from torch.serialization import add_safe_globals
    from fairseq.data.dictionary import Dictionary
    add_safe_globals([Dictionary])
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [f"{lib_dir}/base_model/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
"""

    # Robust replacement using Regex to find the function definition and body
    # Looks for 'def load_hubert(config,lib_dir):' and consumes lines until the next top-level def/class/import or end of file
    # Provided function has simple signature, but indentation handling in regex is tricky.
    # We'll rely on the specific indentation pattern of the file.
    
    pattern = r"def load_hubert\(config,\s*lib_dir\):.*?(?=\ndef |$)"
    # Note: dotall=True makes . match newlines. But we need to be careful not to consume the next function.
    # A safer approach might be to just locate the start and replace until we see a line that starts with specific keywords or dedent.
    # Given the known structure, we can try a slightly looser match or just a block replace if we knew the exact original content.
    # Since we don't know the exact original content, we will use a logic that finds the start line, and finds the end line by indentation.

    lines = content.splitlines()
    start_index = -1
    end_index = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith("def load_hubert(config,lib_dir):") or line.strip().startswith("def load_hubert(config, lib_dir):"):
            start_index = i
            break
            
    if start_index == -1:
        print("[FIX] Could not locate 'def load_hubert' in the file. Already patched?")
        return

    # Find where the function ends (next line with same or lower indentation level, assuming 4 spaces or less, but usually top level functions are 0 indentation if not in class)
    # Checking existing file indentation would be good. Assuming file is standard python.
    # Actually, rvc_python modules might be top level functions.
    
    for i in range(start_index + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            # We hit something with 0 indentation (like a new def or import)
            end_index = i
            break
    
    if end_index == -1:
        end_index = len(lines)

    print(f"[FIX] Replacing lines {start_index} to {end_index}...")

    # Reconstruct content
    new_content_lines = lines[:start_index] + [new_function.strip()] + lines[end_index:]
    new_content = "\n".join(new_content_lines)

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("[FIX] Successfully patched load_hubert.")

if __name__ == "__main__":
    fix_rvc_utils()
