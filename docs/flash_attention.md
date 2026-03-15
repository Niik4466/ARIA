# Flash Attention Configuration

Flash Attention is a performance optimization that can significantly improve inference times and reduce memory usage for advanced models like `qwen3_tts`.

**Note:** Flash Attention is currently only supported on **NVIDIA GPUs**.

## Installation

Installing Flash Attention from source can be very slow and prone to errors. Fortunately, you can install pre-built wheels to easily set it up.

### Step 1: Find the right wheel

1. Go to the official wheel repository: [https://flashattn.dev/](https://flashattn.dev/)
2. Select your platform (Linux or Windows).
3. The site will ask you to select three parameters to generate the correct download link:
   - **Flash Attention Version** (usually pick the latest available)
   - **Python Version**
   - **CUDA & PyTorch Version**

### Step 2: Get your system versions

To know exactly which Python, PyTorch, and CUDA versions you are running, make sure your ARIA virtual environment is activated and run the following command.

**For Linux and Windows:**
```bash
python -c "import torch, sys; print(f'Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')"
```
*This command will output something like: `Python: 3.10.12 | PyTorch: 2.5.1+cu124 | CUDA: 12.4`*

### Step 3: Install inside `aria_venv`

1. Once you have selected the correct options on the website, copy the generated `pip install` command (it should end with a link to a `.whl` file).
2. Make sure you are in the root ARIA directory and your virtual environment is activated:

**Linux:**
```bash
source aria_venv/bin/activate
# Paste the copied command here, for example:
pip install https://.../flash_attn-...whl
```

**Windows:**
```bat
call aria_venv\Scripts\activate
REM Paste the copied command here, for example:
pip install https://.../flash_attn-...whl
```

## Enable in config.py

Once the installation is complete, open `config.py` in the root directory and toggle the `FLASH_ATTENTION` setting to `True`:

```python
# --- TTS ---
USE_QWEN3_TTS = True
QWEN3_LANG = "Spanish"
FLASH_ATTENTION = True   # <--- Change this to True
```

Restart ARIA, and the underlying models will now utilize Flash Attention for faster performance.
