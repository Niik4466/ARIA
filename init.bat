@echo off
setlocal

REM ============================================================================
REM ARIA SYSTEM INITIALIZATION SCRIPT (Windows)
REM ============================================================================

REM 1. Start Ollama Service
echo [INIT] Checking Ollama service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INIT] Ollama is already running.
) else (
    echo [INIT] Starting Ollama in background...
    start /B "Ollama Core" ollama serve
    REM Give it time to initialize
    timeout /t 5 /nobreak >nul
)

REM 2. Pull Required Models
echo [INIT] Pulling required models...

REM Extract models from config.py using python
for /f "tokens=*" %%i in ('python -c "import config; print(config.RESPONSE_MODEL)"') do set RESPONSE_MODEL=%%i
for /f "tokens=*" %%i in ('python -c "import config; print(config.DECISOR_MODEL)"') do set DECISOR_MODEL=%%i

echo [INFO] Pulling RESPONSE_MODEL: %RESPONSE_MODEL%
call ollama pull %RESPONSE_MODEL%

if not "%RESPONSE_MODEL%"=="%DECISOR_MODEL%" (
    echo [INFO] Pulling DECISOR_MODEL: %DECISOR_MODEL%
    call ollama pull %DECISOR_MODEL%
)

echo [INFO] Pulling embedding model: nomic-embed-text
call ollama pull nomic-embed-text



REM 4. Start ARIA Main Application
echo [INIT] Starting ARIA Main Application...
if not exist "aria_venv" (
    echo [ERROR] 'aria_venv' not found. Please run install-windows.bat first.
    pause
    exit /b 1
)

call aria_venv\Scripts\activate
echo [ARIA] Context activated. Application starting...
python main.py

REM Capture exit
if %errorlevel% neq 0 (
    echo [ARIA] Application exited with error.
    pause
)

call deactivate
echo [INIT] System shutdown.
pause
