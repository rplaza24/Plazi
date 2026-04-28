#!/bin/bash
# Installation script for Gimbal Control System
# Run this on your NVIDIA Jetson Orin Nano

set -e

echo "=========================================="
echo "Gimbal Control System Installer"
echo "For Jetson Orin Nano + Insta360 One X2"
echo "=========================================="

# Update package lists
echo "[1/5] Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "[2/5] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-serial \
    python3-smbus \
    i2c-tools \
    libatlas-base-dev \
    libhdf5-serial-dev \
    libhdf5-dev \
    libqtgui4 \
    libqt5test5 \
    libxcb-xinerama0 \
    git

# Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install \
    torch \
    torchvision \
    opencv-python-headless \
    pyserial \
    smbus2 \
    numpy \
    pillow \
    matplotlib

# Optional: Install for PWM servo control
echo "[4/5] Installing optional PWM libraries..."
pip3 install adafruit-circuitpython-pca9685 adafruit-blinka || echo "PWM libraries skipped (optional)"

# Set permissions
echo "[5/5] Setting up permissions..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G i2c $USER
sudo usermod -a -G video $USER

# Create udev rules
echo "Creating udev rules..."
sudo bash -c 'cat > /etc/udev/rules.d/99-gimbal.rules << EOF
KERNEL=="ttyTHS*", MODE="0666"
KERNEL=="i2c-[0-9]*", MODE="0666"
EOF'

sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Reboot your Jetson: sudo reboot"
echo "2. Connect your Insta360 One X2 in UVC mode"
echo "3. Connect your gimbal (see GIMBAL_SETUP_GUIDE.md)"
echo "4. Run calibration: python3 calibrate_gimbal.py --simulation"
echo "5. Start tracking: python3 emotion_tracker.py --simulation"
echo ""
echo "For detailed setup instructions, see:"
echo "  - README.md"
echo "  - GIMBAL_SETUP_GUIDE.md"
echo ""
