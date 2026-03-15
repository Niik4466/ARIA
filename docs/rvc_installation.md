# ARIA RVC Backend Installation Guide

Deploying Retrieval-based Voice Conversion (RVC) alongside ARIA natively can be tricky due to specific dependency legacy constraints, model compatibility architectures, and breaking changes in modern Python tools. 

Following these steps exactly will prevent the "static noise" inferencing bug and dependency mismatches.

## Prerequisites
Ensure that your Python virtual environment (`venv`) is activated for the ARIA project.

```bash
source venv/bin/activate
```

## Step 1: Clone the Official RVC-WebUI Core
ARIA relies on the robust VITS math processing and Overlap-Add (OLA) padding algorithms from the official WebUI repository.

Clone the official UI directly into your `src/rvc` directory:

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI src/rvc/rvc_webui
```

## Step 2: Temporarily Downgrade Pip
The core framework depends heavily on `fairseq==0.12.2`. This package relies on an older version of `omegaconf` which contains a metadata typo (`PyYAML (>=5.1.*)`). 
Modern pip versions (24.1+) enforce strict PEP-440 metadata rules and will refuse to install it, throwing a `Metadata Generation Failed` error.

Downgrade pip to a compatible legacy version to bypass this check:

```bash
pip install pip==23.3.1
```

## Step 3: Install RVC Dependencies
With the downgraded pip, safely install all required packages:

```bash
pip install fairseq==0.12.2 faiss-cpu ffmpeg-python praat-parselmouth pyworld torchcrepe
```

*Note: Once these packages are successfully installed, you can restore pip to its latest version:*
```bash
pip install --upgrade pip
```

## Step 4: Download Base Audio Models
RVC requires two crucial `.pt` base models to process content embeddings and extract pitch contours:
1. **HuBERT** (`hubert_base.pt`): Extracts latent audio embeddings matching the RVC V2 latent space. (Using alternative implementations like HuggingFace `transformers` will result in pure static noise output).
2. **RMVPE** (`rmvpe.pt`): Extracts precise pitch (F0) tracking.

The WebUI includes a downloader script for these:
```bash
python src/rvc/rvc_webui/tools/download_models.py
```
This script will download `hubert_base.pt` to `src/rvc/rvc_webui/assets/hubert/` and `rmvpe.pt` to `src/rvc/rvc_webui/assets/rmvpe/`.

## Step 5: Place Voice Models
Put your specific RVC voice models inside the `models/rvc/checkpoints/` folder.
Both the `.pth` weights file and `.index` FAISS file should be placed together here.

```
ARIA/
└── models/
    └── rvc/
        └── checkpoints/
            ├── MikuAI.pth
            └── MikuAI.index
```

## Note on PyTorch 2.6+ Security (Weights Only)
In PyTorch versions >= 2.6, `torch.load` defaults to `weights_only=True` to prevent arbitrary code execution from malicious pickles. 

Since RVC weights and the `fairseq` dictionaries serialize complex Python objects, this security feature will crash the engine on load. 

ARIA's local `src/rvc/inference.py` orchestrator applies a direct monkeypatch to disable this restriction:
```python
# Hotfix for PyTorch 2.6+ in inference.py
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
```
**Warning:** Because of this bypass, **never** load untrusted `.pth` RVC checkpoints downloaded from unverified sources, as they could execute malicious code on your machine upon loading. Only use trustworthy or custom-trained models.
