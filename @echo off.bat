@echo off
set VENV_NAME=gemma_finetune_env

echo 🚀 Starting environment setup for FunctionGemma fine-tuning on Windows...

:: 1. Create Virtual Environment
if not exist %VENV_NAME% (
    echo 📂 Creating virtual environment: %VENV_NAME%...
    python -m venv %VENV_NAME%
) else (
    echo ✅ Virtual environment already exists.
)

:: 2. Activate the Environment
echo 🔌 Activating environment...
call %VENV_NAME%\Scripts\activate

:: 3. Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install Core Dependencies
echo 📦 Installing required packages for RTX 40-series...
:: Installing PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 ^
            datasets ^
            accelerate ^
            peft ^
            trl ^
            bitsandbytes ^
            scipy ^
            tensorboard

:: 5. Verify Installation
echo 🔍 Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -------------------------------------------------------
echo ✅ Setup Complete!
echo To start your environment, run: %VENV_NAME%\Scripts\activate
echo Then you can run your script: python finetune_functiongemma.py
echo -------------------------------------------------------
pause