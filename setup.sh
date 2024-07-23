#!/usr/bin/bash

set +x
set -e

deactivate || true
rm -rf .env

python3 -m venv .env
source .env/bin/activate


#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip3 install torch-directml
pip3 install diffusers["torch"] transformers accelerate matplotlib


