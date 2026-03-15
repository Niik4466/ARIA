@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM ARIA & RVC INSTALLATION SCRIPT (Windows)
REM ============================================================================

set GPU_MODE=
set GPU_MODEL=
set INSTALL_RVC=0

:parse_args
if "%~1"=="" goto main
if "%~1"=="--help" goto usage
if "%~1"=="-h" goto usage
if "%~1"=="--gpu" (
    set GPU_MODE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--gpu_model" (
    set GPU_MODEL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--rvc" (
    set INSTALL_RVC=1
    shift
    goto parse_args
)
shift
goto parse_args

:main
if "%GPU_MODE%"=="" (
    echo [ERROR] --gpu argument is required.
    goto usage
)

echo [INSTALL] Starting Installation Sequence...
echo [INFO] GPU_MODE: %GPU_MODE%
echo [INFO] GPU_MODEL: %GPU_MODEL%

REM ============================================================================
REM PHASE 1: ARIA (Main System)
REM ============================================================================
echo.
echo [PHASE 1] Setting up ARIA environment...

if not exist aria_venv (
    echo [ARIA] Creating venv 'aria_venv'...
    python -m venv aria_venv
) else (
    echo [ARIA] 'aria_venv' already exists.
)

echo [ARIA] Activating environment...
call aria_venv\Scripts\activate

echo [ARIA] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install ARIA dependencies.
    goto end
)


echo [ARIA] Configuring PyTorch for %GPU_MODE%...

set INSTALLED_TORCH=0

if /I "%GPU_MODE%"=="cuda" (
    REM Check if GPU_MODEL contains "50" (e.g. 5070, 5080, 5090)
    echo "%GPU_MODEL%" | findstr "50" >nul
    if !errorlevel! equ 0 (
        echo [TORCH] Detected RTX 50-series. Installing CUDA 12.8 compatible torch...
        pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
        set INSTALLED_TORCH=1
    ) else (
        echo [TORCH] Standard CUDA GPU. Installing default torch...
        pip install torch torchvision
        set INSTALLED_TORCH=1
    )
)

if /I "%GPU_MODE%"=="rocm" (
    echo [WARNING] Native ROCm is not supported on Windows.
    echo [TORCH] Installing torch-directml instead...
    pip install --upgrade torch-directml
    set INSTALLED_TORCH=1
)

REM Default case if not handled above
if "!INSTALLED_TORCH!"=="0" (
    echo [TORCH] Installing standard torch versions...
    pip install torch torchvision
)

echo [ARIA] Deactivating...
call deactivate

REM ============================================================================
REM PHASE 2: RVC (Subsystem)
REM ============================================================================

if "%INSTALL_RVC%"=="1" (
    echo.
    echo [PHASE 2] Setting up RVC Subsystem...
    
    echo [ARIA] Activating environment...
    call aria_venv\Scripts\activate
    
    echo [RVC] Cloning official WebUI...
    if not exist "src\rvc\rvc_webui" (
        git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI src\rvc\rvc_webui
    ) else (
        echo [RVC] 'src\rvc\rvc_webui' already exists. Skipping clone.
    )
    
    echo [CRITICAL] Downgrading pip for fairseq compatibility...
    python -m pip install pip==23.3.1
    
    echo [RVC] Installing RVC dependencies...
    pip install fairseq==0.12.2 faiss-cpu ffmpeg-python praat-parselmouth pyworld torchcrepe
    
    echo [RVC] Upgrading pip back...
    python -m pip install --upgrade pip
    
    echo [RVC] Downloading base models...
    python src\rvc\rvc_webui\tools\download_models.py
    
    echo [ARIA] Deactivating...
    call deactivate
) else (
    echo.
    echo [INFO] Skipping RVC Installation (--rvc flag not provided).
)

echo.
echo [INSTALL] Installation Complete!
echo.
goto end

:usage
echo.
echo Usage: %~nx0 --gpu ^<gpu_mode^> [--gpu_model ^<model^>]
echo.
echo Options:
echo   --gpu ^<mode^>       Specify GPU mode: "cuda" (NVIDIA) or "rocm" (AMD).
echo   --gpu_model ^<model^> Optional. Specify GPU model (e.g., 5090) for specific handling.
echo   --rvc              Optional. Install RVC backend alongside ARIA.
echo   --help             Show this help message.
echo.

:end
pause
