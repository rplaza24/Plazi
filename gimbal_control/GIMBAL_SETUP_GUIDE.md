# Gimbal Setup Guide

Complete guide for setting up emotion-driven gimbal tracking with Insta360 One X2 and NVIDIA Jetson Orin Nano.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Gimbal Connection Options](#gimbal-connection-options)
3. [Wiring Diagrams](#wiring-diagrams)
4. [Software Installation](#software-installation)
5. [Calibration](#calibration)
6. [Testing & Tuning](#testing--tuning)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Required Components

- **NVIDIA Jetson Orin Nano** (4GB or 8GB)
- **Insta360 One X2** camera (in UVC mode)
- **Pan-Tilt Gimbal** (see compatible models below)
- **MicroSD card** (32GB+, for Jetson OS)
- **Power supply** (adequate for Jetson + gimbal)
- **USB-C cable** (camera to Jetson)
- **Jumper wires** (for serial/PWM connection)

### Compatible Gimbals

#### Serial-Controlled (Recommended)
- **DFRobot 2-Axis Gimbal** (UART protocol)
- **Feetech FS90R** continuous servo with controller
- **Custom build** with Arduino/ESP32 as intermediary

#### PWM-Controlled
- **MG996R servos** (2x for pan/tilt)
- **PCA9685 PWM driver** board
- **Tower Pro SG90** (lightweight cameras)

#### USB-Controlled
- **DJI Ronin-SC** (requires SDK)
- **Zhiyun Crane M2** (requires cloud service)

---

## Gimbal Connection Options

### Option 1: UART/Serial (Most Common)

**Best for:** Custom gimbals, DFRobot, Arduino-based systems

**Jetson Orin Nano UART Pins:**
```
Pin 8 (GPIO8):  TXD0 (transmit)
Pin 10 (GPIO9): RXD0 (receive)
Pin 6:          GND
```

**Wiring:**
```
Jetson TXD0  →  Gimbal RX
Jetson RXD0  →  Gimbal TX
Jetson GND   →  Gimbal GND
```

**Enable UART on Jetson:**
```bash
# Edit boot configuration
sudo nano /boot/extlinux/extlinux.conf

# Add to APPEND line:
console=ttyS0,115200n8 console=tty0

# Enable serial console
sudo systemctl disable nvgetty.service

# Reboot
sudo reboot
```

**Test connection:**
```bash
# Install screen
sudo apt-get install screen

# Connect to gimbal
screen /dev/ttyTHS1 115200

# Send test command (adjust for your protocol)
echo "0,0" > /dev/ttyTHS1
```

### Option 2: PWM via PCA9685

**Best for:** Multiple servos, precise control

**Wiring:**
```
Jetson I2C SDA (Pin 3)  →  PCA9685 SDA
Jetson I2C SCL (Pin 5)  →  PCA9685 SCL
Jetson 5V (Pin 2/4)     →  PCA9685 VCC
Jetson GND (Pin 6)      →  PCA9685 GND

PCA9685 CH0  →  Pan servo signal
PCA9685 CH1  →  Tilt servo signal
PCA9685 V+   →  External 5V/6V power
External GND →  Jetson GND (common ground)
```

**Enable I2C:**
```bash
# Load I2C modules
sudo modprobe i2c-dev

# Make permanent
echo "i2c-dev" | sudo tee -a /etc/modules

# Scan for devices
i2cdetect -y -r 0
```

**Install PWM library:**
```bash
pip3 install adafruit-circuitpython-pca9685
pip3 install adafruit-blinka
```

### Option 3: USB Gimbal

**Best for:** DJI, Zhiyun commercial gimbals

**Setup:**
```bash
# Check USB detection
lsusb

# Install manufacturer SDK
# DJI: https://developer.dji.com/onboard-sdk/
# Zhiyun: https://www.zhiyun-tech.com/api/en
```

---

## Software Installation

### Step 1: Install System Dependencies

```bash
sudo apt-get update
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
    libqt5test5
```

### Step 2: Install Python Dependencies

```bash
pip3 install \
    torch \
    torchvision \
    opencv-python-headless \
    pyserial \
    smbus2 \
    numpy \
    pillow
```

### Step 3: Verify Installation

```bash
# Test camera
v4l2-ctl --list-devices

# Test serial port
ls -la /dev/ttyTHS*

# Test I2C (if using PWM)
i2cdetect -y -r 0
```

---

## Calibration

### Quick Calibration

```bash
# Run in simulation mode first (no hardware)
python3 calibrate_gimbal.py --simulation

# Then with actual hardware
python3 calibrate_gimbal.py --config gimbal_config.json
```

### Calibration Steps

1. **Manual Control Test**
   ```bash
   python3 calibrate_gimbal.py --mode manual
   ```
   - Use arrow keys to test movement
   - Verify direction matches expectations
   - Check range limits

2. **Auto Calibration**
   ```bash
   python3 calibrate_gimbal.py --mode auto
   ```
   - Gimbal moves to predefined positions
   - Verify each position is accurate
   - Adjust limits if needed

3. **PID Tuning**
   ```bash
   python3 calibrate_gimbal.py --mode pid
   ```
   - Start with Kp=0.5, Ki=0.0, Kd=0.1
   - Increase Kp until slight oscillation
   - Reduce Kp by 50%
   - Add Kd to dampen oscillations
   - Add small Ki to eliminate steady-state error

### Coordinate Mapping

The system maps image coordinates to gimbal angles:

```
Image Center (320, 240) → Gimbal (0°, 0°)
Image Left Edge (0, 240) → Gimbal (-90°, 0°)
Image Right Edge (640, 240) → Gimbal (+90°, 0°)
Image Top (320, 0) → Gimbal (0°, +45°)
Image Bottom (320, 480) → Gimbal (0°, -45°)
```

Adjust multipliers in `image_coords_to_angles()` based on your:
- Camera field of view (FOV)
- Gimbal mechanical limits
- Desired tracking sensitivity

---

## Testing & Tuning

### Test Without Camera

```bash
# Simulated tracking
python3 emotion_tracker.py --simulation
```

### Test With Camera

```bash
# Basic tracking
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1

# Track only happy faces
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1 \
    --target-emotion happy \
    --confidence-threshold 0.7

# Headless mode (embedded)
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1 \
    --headless \
    --log-file tracking.log
```

### Performance Tuning

**For smoother tracking:**
- Increase `smoothing_factor` (0.7 → 0.85)
- Increase `deadzone` (5 → 10 pixels)
- Reduce `max_speed` (30 → 20 deg/s)

**For faster response:**
- Increase `kp` (0.8 → 1.2)
- Decrease `smoothing_factor` (0.7 → 0.5)
- Reduce `deadzone` (5 → 2 pixels)

**For better accuracy:**
- Tune PID parameters carefully
- Ensure good lighting
- Use higher resolution camera feed
- Train custom emotion model on your dataset

---

## Troubleshooting

### Gimbal Not Moving

**Check connections:**
```bash
# Verify serial port exists
ls -la /dev/ttyTHS*

# Check permissions
sudo chmod 666 /dev/ttyTHS1

# Test with screen
screen /dev/ttyTHS1 115200
# Type commands manually
```

**Verify protocol:**
- Check gimbal documentation for correct baud rate
- Confirm TX/RX wiring (should be crossed)
- Ensure common ground connection

### Jittery Movement

**Solutions:**
1. Reduce Kp (proportional gain)
2. Increase Kd (derivative gain)
3. Increase deadzone parameter
4. Increase smoothing_factor
5. Check power supply stability
6. Ensure camera frame rate is consistent

### Lost Tracking

**Solutions:**
1. Improve lighting conditions
2. Adjust confidence_threshold (lower to 0.5)
3. Reduce min_face_size
4. Clean camera lens
5. Check camera focus
6. Verify face is within detectable range

### Camera Issues

**Insta360 One X2 not detected:**
```bash
# Check USB connection
lsusb

# Verify UVC mode
v4l2-ctl --list-devices

# Reset camera
# On camera: Settings > Advanced > USB Mode > UVC
```

### Permission Errors

```bash
# Add user to dialout group (serial)
sudo usermod -a -G dialout $USER

# Add user to i2c group
sudo usermod -a -G i2c $USER

# Create udev rules
sudo nano /etc/udev/rules.d/99-gimbal.rules

# Add:
KERNEL=="ttyTHS*", MODE="0666"
BUS=="usb", ATTRS{idVendor}=="xxxx", MODE="0666"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Example Configurations

### Fast Tracking (for quick movements)

```json
{
  "tracking": {
    "pid_pan": {"kp": 1.2, "ki": 0.02, "kd": 0.2},
    "pid_tilt": {"kp": 1.2, "ki": 0.02, "kd": 0.2},
    "deadzone": 3,
    "max_speed": 60,
    "smoothing_factor": 0.5
  }
}
```

### Smooth Cinematic (for video)

```json
{
  "tracking": {
    "pid_pan": {"kp": 0.5, "ki": 0.005, "kd": 0.25},
    "pid_tilt": {"kp": 0.5, "ki": 0.005, "kd": 0.25},
    "deadzone": 15,
    "max_speed": 15,
    "smoothing_factor": 0.85
  }
}
```

### Balanced (general purpose)

```json
{
  "tracking": {
    "pid_pan": {"kp": 0.8, "ki": 0.01, "kd": 0.15},
    "pid_tilt": {"kp": 0.8, "ki": 0.01, "kd": 0.15},
    "deadzone": 5,
    "max_speed": 30,
    "smoothing_factor": 0.7
  }
}
```

---

## Next Steps

After successful calibration:

1. **Train custom model** on your specific use case
2. **Optimize for TensorRT** for maximum FPS
3. **Add multi-person tracking** with ID persistence
4. **Implement gesture recognition** for manual override
5. **Create web interface** for remote monitoring
6. **Add logging and analytics** for behavior analysis

See the main README.md for integration examples and advanced features.
