#!/bin/bash
# Install dependencies for training emotion recognition model

echo "Installing training dependencies..."

# Update system
sudo apt-get update

# Install system dependencies
sudo apt-get install -y python3-pip python3-dev

# Install Python packages
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
pip3 install pandas matplotlib scikit-learn
pip3 install opencv-python-headless

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "✓ Training dependencies installed!"
