#!/bin/bash
# Installation script for Insta360 One X2 Emotion Recognition System
# Run this on your NVIDIA Jetson Orin Nano

set -e

echo "=========================================="
echo "Insta360 One X2 Emotion Recognition Setup"
echo "=========================================="
echo ""

# Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -y

# Install OpenCV and video utilities
echo "[2/5] Installing OpenCV and dependencies..."
sudo apt-get install -y python3-pip python3-opencv libopencv-dev
sudo apt-get install -y v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-good
sudo apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-libav

# Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy torch torchvision

# For Jetson, use pre-built PyTorch with CUDA support
# If the above doesn't work, try:
# pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

# Install additional ML libraries (optional)
echo "[4/5] Installing optional ML libraries..."
pip3 install transformers pillow opencv-python-headless

# Verify installation
echo "[5/5] Verifying installation..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect your Insta360 One X2 via USB-C"
echo "2. Set camera to UVC mode (Settings > Advanced > USB Mode > UVC)"
echo "3. Verify camera detection: v4l2-ctl --list-devices"
echo "4. Run the emotion recognition: python3 emotion_recognition.py"
echo ""
echo "For RTSP streaming alternative, see README.md"
echo ""
