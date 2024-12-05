#!/bin/bash
echo -e "installing PyTorch"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
echo -e " installing required packages" 
pip install -r requirements.txt
pip install git+https://github.com/kunaldahiya/pyxclib
pip install dill --upgrade
echo -e " Environment setup finished successfuly"