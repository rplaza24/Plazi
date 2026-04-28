# Insta360 One X2 + Jetson Orin Nano Emotion Recognition Setup

## Overview
This guide shows how to connect an Insta360 One X2 camera to an NVIDIA Jetson Orin Nano and build an emotion recognition system.

## Hardware Requirements
- NVIDIA Jetson Orin Nano
- Insta360 One X2 camera
- USB-C cable (for direct connection) OR WiFi network
- MicroSD card in the camera

## Connection Methods

### Method 1: USB Direct Connection (Recommended)
The Insta360 One X2 can connect via USB as a UVC (USB Video Class) device.

1. **Enable UVC Mode on Insta360 One X2:**
   - Power on the camera
   - Swipe down to access settings
   - Go to Settings > Advanced > USB Mode
   - Select "UVC Mode" or "Webcam Mode"

2. **Connect via USB-C:**
   - Connect the camera to Jetson Orin Nano using USB-C cable
   - The camera should appear as `/dev/video0` or similar

3. **Verify Connection:**
   ```bash
   ls -la /dev/video*
   v4l2-ctl --list-devices
   ```

### Method 2: WiFi Streaming (RTSP)
If USB doesn't work reliably, use RTSP streaming over WiFi.

1. **Connect both devices to same WiFi network**

2. **Enable RTSP on Insta360:**
   - The One X2 supports RTSP streaming through the Insta360 app
   - Note: RTSP may require firmware updates or specific settings

3. **Alternative: Use ADB (Android Debug Bridge)**
   ```bash
   # Install ADB on Jetson
   sudo apt-get install android-tools-adb
   
   # Connect via WiFi (camera IP)
   adb connect <camera_ip>:8888
   
   # Stream using screen capture or camera API
   ```

## Software Setup

### 1. Install Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv libopencv-dev
sudo apt-get install -y v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-good
```

### 2. Python Environment
```bash
pip3 install opencv-python numpy torch torchvision
pip3 install transformers pillow
```

### 3. Verify Camera Detection
```bash
# List video devices
v4l2-ctl --list-devices

# Test video stream
gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink
```

## Emotion Recognition Pipeline

The system will:
1. Capture video from Insta360 One X2
2. Extract faces from each frame
3. Run emotion recognition model
4. Display/Log results

## Troubleshooting

### Camera Not Detected
- Try different USB-C cables
- Ensure camera is in UVC mode
- Check power supply (camera may need external power)
- Try: `sudo modprobe uvcvideo`

### Low Frame Rate
- Reduce resolution: `v4l2-ctl --set-fmt-video=width=1280,height=720`
- Use hardware acceleration with GStreamer
- Optimize model inference (use TensorRT)

### RTSP Issues
- Verify firewall settings
- Check network latency
- Consider using lower resolution streams

## Next Steps
See `emotion_recognition.py` for the complete implementation code.
