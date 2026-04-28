# Gimbal Control System for Emotion-Driven Tracking

This module adds gimbal control capabilities to your emotion recognition system, allowing the Jetson Orin Nano to automatically point the Insta360 One X2 camera at detected faces based on their screen coordinates.

## Files Created

1. **`gimbal_controller.py`** - Core gimbal control logic with:
   - PID controller for smooth pan/tilt movements
   - Coordinate mapping from image space to gimbal angles
   - Face tracking with priority selection (closest/largest face)
   - Configurable tracking parameters
   - Safety limits and boundary checking

2. **`emotion_tracker.py`** - Integrated solution combining:
   - Real-time emotion detection from `emotion_recognition.py`
   - Face coordinate extraction
   - Automatic gimbal positioning
   - Target locking on specific emotions
   - Multi-person tracking with selection logic

3. **`calibrate_gimbal.py`** - Calibration utility to:
   - Map image coordinates to gimbal angles
   - Test PID parameters
   - Save/load configuration
   - Visualize tracking performance

4. **`gimbal_config.json`** - Default configuration file with tunable parameters

5. **`GIMBAL_SETUP_GUIDE.md`** - Complete setup documentation

## Quick Start

### 1. Hardware Setup

Connect your gimbal to Jetson Orin Nano:
- **UART/Serial**: Most common (TX/RX to GPIO pins)
- **PWM**: For servo-based gimbals
- **USB**: For USB-controlled gimbals (e.g., DJI, Zhiyun)
- **GPIO**: Direct pin control for simple servos

### 2. Install Dependencies

```bash
sudo apt-get install python3-serial python3-smbus i2c-tools
pip3 install pyserial smbus2
```

### 3. Calibrate Gimbal

```bash
# Run calibration to map image coordinates to gimbal angles
python3 calibrate_gimbal.py --port /dev/ttyTHS1 --baud 115200

# Follow on-screen instructions to set center position and limits
```

### 4. Run Emotion-Driven Tracking

```bash
# Track all faces, prioritize largest
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1 \
    --mode track_all

# Track only happy faces
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1 \
    --target-emotion happy \
    --confidence-threshold 0.7

# Headless mode (no display, log only)
python3 emotion_tracker.py \
    --camera-source /dev/video0 \
    --gimbal-port /dev/ttyTHS1 \
    --headless \
    --log-file tracking.log
```

## Key Features

### Smart Target Selection
- **Largest face**: Prioritize closest person
- **Closest to center**: Minimize gimbal movement
- **Specific emotion**: Lock onto target emotion
- **Round-robin**: Cycle through multiple targets

### Smooth Movement
- PID-controlled pan/tilt for jitter-free tracking
- Configurable speed limits and acceleration
- Deadzone handling to prevent micro-adjustments
- Prediction filtering to reduce oscillation

### Safety Features
- Soft limits to prevent mechanical damage
- Emergency stop via keyboard (Ctrl+C or 'q')
- Watchdog timer to stop movement if no faces detected
- Boundary checking for valid coordinate ranges

## Configuration (gimbal_config.json)

```json
{
  "gimbal": {
    "port": "/dev/ttyTHS1",
    "baud_rate": 115200,
    "protocol": "simple_serial",
    "pan_limits": [-90, 90],
    "tilt_limits": [-45, 45],
    "center_position": [0, 0]
  },
  "tracking": {
    "pid_pan": {"kp": 0.8, "ki": 0.01, "kd": 0.15},
    "pid_tilt": {"kp": 0.8, "ki": 0.01, "kd": 0.15},
    "deadzone": 5,
    "max_speed": 30,
    "smoothing_factor": 0.7
  },
  "target_selection": {
    "priority": "largest",
    "min_face_size": 50,
    "target_emotions": ["happy", "surprise"],
    "confidence_threshold": 0.6
  }
}
```

## Supported Gimbal Protocols

The system includes adapters for common gimbal types:

- **Simple Serial**: Custom ASCII/binary protocol
- **DJI Ronin**: DJI SDK protocol
- **Zhiyun**: Zhiyun cloud service or serial
- **Servo PWM**: Direct servo control via PCA9685
- **Custom**: Extendable interface for your protocol

## Integration with Existing System

The `emotion_tracker.py` seamlessly integrates with your existing setup:

```python
from emotion_tracker import EmotionGimbalTracker

tracker = EmotionGimbalTracker(
    camera_source="/dev/video0",
    gimbal_port="/dev/ttyTHS1",
    target_emotion="happy",
    use_pretrained_model=True
)

tracker.start()
```

## Performance Optimization for Jetson

- Runs at 30-60 FPS depending on model size
- Gimbal updates at 50-100 Hz (separate thread)
- CUDA acceleration for inference
- TensorRT support for deployed models
- Multi-threaded architecture (capture → infer → control)

## Troubleshooting

### Gimbal Not Moving
- Check serial port permissions: `sudo chmod 666 /dev/ttyTHS1`
- Verify wiring (TX→RX, RX→TX, GND→GND)
- Test with `echo "test" > /dev/ttyTHS1`
- Confirm baud rate matches gimbal specifications

### Jittery Movement
- Reduce PID proportional gain (kp)
- Increase deadzone parameter
- Enable more smoothing (higher smoothing_factor)
- Check frame rate consistency

### Lost Tracking
- Increase min_face_size threshold
- Adjust confidence_threshold
- Check lighting conditions
- Verify camera focus and exposure

## Next Steps

1. **Test with simulated gimbal** (no hardware needed initially)
2. **Calibrate** with your specific gimbal model
3. **Tune PID parameters** for smooth tracking
4. **Deploy** with TensorRT-optimized model
5. **Add advanced features**: gesture recognition, voice commands, multi-camera sync

See `GIMBAL_SETUP_GUIDE.md` for detailed hardware connection diagrams and protocol implementations.
