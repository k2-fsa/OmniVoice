@echo off
setlocal

:: Define local paths for 100% portability
set "OMNIVOICE_ROOT=%~dp0"
set "UV_CACHE_DIR=%OMNIVOICE_ROOT%.uv_cache"
set "UV_PYTHON_INSTALL_DIR=%OMNIVOICE_ROOT%.uv_python"
set "HF_HOME=%OMNIVOICE_ROOT%.hf_cache"
set "XDG_CACHE_HOME=%OMNIVOICE_ROOT%.cache"
set "PYTHONPATH=%OMNIVOICE_ROOT%"
set "UV_LINK_MODE=copy"

:: Explicitly target the first NVIDIA GPU
set "CUDA_VISIBLE_DEVICES=0"

:: Ensure we are in the correct directory
cd /d "%OMNIVOICE_ROOT%"

echo.
echo ========================================================
echo   OmniVoice Portable Launcher (GPU Enhanced)
echo ========================================================
echo.
echo [1/4] Checking for 'uv' installation...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: 'uv' is not installed or not in PATH.
    echo Please install it from: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo [2/4] Synchronizing dependencies (CUDA 12.8)...
echo (This avoids files scattered outside this folder. First GPU sync will take time.)
uv sync --all-extras

echo [3/4] Verifying GPU/CUDA Status...
uv run python -c "import torch; available = torch.cuda.is_available(); print(f' - Detected GPU: {torch.cuda.get_device_name(0)}' if available else ' - !!! GPU NOT DETECTED !!! (Running on CPU)'); print(f' - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if available else '')"

echo [4/4] Launching OmniVoice Web UI...
uv run omnivoice-demo

echo.
echo Application closed.
pause
