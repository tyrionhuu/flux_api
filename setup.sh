conda create -n flux_api python=3.12.11 -y && conda activate flux_api
git checkout feat/concurrency
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl
pip install fastapi uvicorn opencv-python-headless scipy scikit-image matplotlib aiofiles python-multipart redis celery einops xformers peft nvitop gpustat
