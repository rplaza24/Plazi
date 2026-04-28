#!/usr/bin/env python3
"""
Gimbal Controller Module
Controls pan/tilt gimbal based on face coordinates from emotion detection.
Supports multiple protocols: Serial, PWM, USB, and custom interfaces.
"""

import time
import json
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading

# Try to import serial, make it optional for testing
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GimbalProtocol(Enum):
    """Supported gimbal communication protocols."""
    SIMPLE_SERIAL = "simple_serial"
    DJI_RONIN = "dji_ronin"
    ZHIYUN = "zhiyun"
    SERVO_PWM = "servo_pwm"
    CUSTOM = "custom"
    SIMULATION = "simulation"  # For testing without hardware


@dataclass
class PIDConfig:
    """PID controller configuration."""
    kp: float = 0.8
    ki: float = 0.01
    kd: float = 0.15
    
    def to_dict(self) -> Dict[str, float]:
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PIDConfig':
        return cls(
            kp=data.get("kp", 0.8),
            ki=data.get("ki", 0.01),
            kd=data.get("kd", 0.15)
        )


@dataclass
class GimbalConfig:
    """Complete gimbal configuration."""
    port: str = "/dev/ttyTHS1"
    baud_rate: int = 115200
    protocol: GimbalProtocol = GimbalProtocol.SIMPLE_SERIAL
    pan_limits: Tuple[float, float] = (-90.0, 90.0)
    tilt_limits: Tuple[float, float] = (-45.0, 45.0)
    center_position: Tuple[float, float] = (0.0, 0.0)
    pid_pan: PIDConfig = None
    pid_tilt: PIDConfig = None
    deadzone: float = 5.0  # pixels
    max_speed: float = 30.0  # degrees per second
    smoothing_factor: float = 0.7
    image_width: int = 640
    image_height: int = 480
    
    def __post_init__(self):
        if self.pid_pan is None:
            self.pid_pan = PIDConfig()
        if self.pid_tilt is None:
            self.pid_tilt = PIDConfig()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GimbalConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        gimbal_data = data.get("gimbal", {})
        tracking_data = data.get("tracking", {})
        
        return cls(
            port=gimbal_data.get("port", "/dev/ttyTHS1"),
            baud_rate=gimbal_data.get("baud_rate", 115200),
            protocol=GimbalProtocol(gimbal_data.get("protocol", "simple_serial")),
            pan_limits=tuple(gimbal_data.get("pan_limits", [-90, 90])),
            tilt_limits=tuple(gimbal_data.get("tilt_limits", [-45, 45])),
            center_position=tuple(gimbal_data.get("center_position", [0, 0])),
            pid_pan=PIDConfig.from_dict(tracking_data.get("pid_pan", {})),
            pid_tilt=PIDConfig.from_dict(tracking_data.get("pid_tilt", {})),
            deadzone=tracking_data.get("deadzone", 5.0),
            max_speed=tracking_data.get("max_speed", 30.0),
            smoothing_factor=tracking_data.get("smoothing_factor", 0.7)
        )
    
    def to_file(self, filepath: str):
        """Save configuration to JSON file."""
        data = {
            "gimbal": {
                "port": self.port,
                "baud_rate": self.baud_rate,
                "protocol": self.protocol.value,
                "pan_limits": list(self.pan_limits),
                "tilt_limits": list(self.tilt_limits),
                "center_position": list(self.center_position)
            },
            "tracking": {
                "pid_pan": self.pid_pan.to_dict(),
                "pid_tilt": self.pid_tilt.to_dict(),
                "deadzone": self.deadzone,
                "max_speed": self.max_speed,
                "smoothing_factor": self.smoothing_factor
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class PIDController:
    """PID controller for smooth gimbal movement."""
    
    def __init__(self, config: PIDConfig):
        self.kp = config.kp
        self.ki = config.ki
        self.kd = config.kd
        self.reset()
    
    def reset(self):
        """Reset PID state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error: float) -> float:
        """Calculate PID output for given error."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.001  # Prevent division by zero
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral))  # Clamp integral
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        self.previous_error = error
        self.last_time = current_time
        
        return p_term + i_term + d_term


class GimbalController:
    """Main gimbal controller class."""
    
    def __init__(self, config: GimbalConfig):
        self.config = config
        self.pan_pid = PIDController(config.pid_pan)
        self.tilt_pid = PIDController(config.pid_tilt)
        self.current_pan = config.center_position[0]
        self.current_tilt = config.center_position[1]
        self.target_pan = config.center_position[0]
        self.target_tilt = config.center_position[1]
        self.serial_conn = None
        self.running = False
        self.control_thread = None
        self.position_history = []
        self.smoothed_pan = config.center_position[0]
        self.smoothed_tilt = config.center_position[1]
        
        # Initialize connection based on protocol
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize communication with gimbal."""
        if self.config.protocol == GimbalProtocol.SIMULATION:
            logger.info("Initialized in SIMULATION mode (no hardware)")
            return
        
        if self.config.protocol == GimbalProtocol.SIMPLE_SERIAL:
            if not SERIAL_AVAILABLE:
                logger.warning("pyserial not installed, falling back to simulation")
                self.config.protocol = GimbalProtocol.SIMULATION
                return
            
            try:
                self.serial_conn = serial.Serial(
                    port=self.config.port,
                    baudrate=self.config.baud_rate,
                    timeout=0.1
                )
                logger.info(f"Connected to gimbal on {self.config.port}")
            except Exception as e:
                logger.warning(f"Failed to connect to gimbal: {e}. Using simulation mode.")
                self.config.protocol = GimbalProtocol.SIMULATION
    
    def _send_command(self, pan_angle: float, tilt_angle: float):
        """Send command to gimbal hardware."""
        if self.config.protocol == GimbalProtocol.SIMULATION:
            logger.debug(f"[SIM] Moving to pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")
            self.current_pan = pan_angle
            self.current_tilt = tilt_angle
            return
        
        if self.config.protocol == GimbalProtocol.SIMPLE_SERIAL:
            # Simple ASCII protocol: "PAN,TILT\n"
            # Customize this for your specific gimbal protocol
            command = f"{pan_angle:.2f},{tilt_angle:.2f}\n"
            try:
                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.write(command.encode())
                    logger.debug(f"Sent command: {command.strip()}")
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
        
        # Add other protocol implementations here (DJI, Zhiyun, PWM, etc.)
    
    def image_coords_to_angles(self, face_x: float, face_y: float, 
                                face_w: float, face_h: float) -> Tuple[float, float]:
        """
        Convert face coordinates in image space to gimbal angles.
        
        Args:
            face_x, face_y: Center of face in image coordinates
            face_w, face_h: Width and height of face bounding box
        
        Returns:
            (pan_angle, tilt_angle) offset from center position
        """
        # Calculate offset from image center
        image_center_x = self.config.image_width / 2
        image_center_y = self.config.image_height / 2
        
        x_offset = face_x - image_center_x
        y_offset = face_y - image_center_y
        
        # Normalize to [-1, 1] range
        x_normalized = x_offset / image_center_x
        y_normalized = y_offset / image_center_y
        
        # Convert to angles based on field of view
        # Adjust these multipliers based on your camera FOV and gimbal range
        pan_fov_multiplier = (self.config.pan_limits[1] - self.config.pan_limits[0]) / 2
        tilt_fov_multiplier = (self.config.tilt_limits[1] - self.config.tilt_limits[0]) / 2
        
        pan_angle = x_normalized * pan_fov_multiplier
        tilt_angle = -y_normalized * tilt_fov_multiplier  # Negative because Y is inverted
        
        return pan_angle, tilt_angle
    
    def select_target_face(self, faces: list, priority: str = "largest",
                          target_emotions: list = None,
                          confidence_threshold: float = 0.6) -> Optional[dict]:
        """
        Select the best face to track from multiple detections.
        
        Args:
            faces: List of face dictionaries with keys:
                   - bbox: (x, y, w, h)
                   - emotion: string
                   - confidence: float
            priority: Selection strategy ("largest", "closest_to_center", "specific_emotion")
            target_emotions: List of emotions to prioritize
            confidence_threshold: Minimum confidence to consider
        
        Returns:
            Selected face dictionary or None
        """
        # Filter by confidence
        valid_faces = [f for f in faces if f.get("confidence", 0) >= confidence_threshold]
        
        if not valid_faces:
            return None
        
        # Filter by emotion if specified
        if target_emotions:
            emotion_faces = [f for f in valid_faces 
                           if f.get("emotion", "").lower() in [e.lower() for e in target_emotions]]
            if emotion_faces:
                valid_faces = emotion_faces
        
        if priority == "largest":
            # Select largest face (by area)
            return max(valid_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        
        elif priority == "closest_to_center":
            # Select face closest to image center
            image_center = (self.config.image_width / 2, self.config.image_height / 2)
            def distance_to_center(face):
                x, y = face["bbox"][0] + face["bbox"][2]/2, face["bbox"][1] + face["bbox"][3]/2
                return ((x - image_center[0])**2 + (y - image_center[1])**2)**0.5
            return min(valid_faces, key=distance_to_center)
        
        elif priority == "specific_emotion" and target_emotions:
            # Already filtered by emotion, just pick largest
            return max(valid_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        
        # Default: largest face
        return max(valid_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
    
    def update_target(self, face_bbox: Tuple[int, int, int, int], 
                     use_pid: bool = True) -> Tuple[float, float]:
        """
        Update gimbal target position based on face location.
        
        Args:
            face_bbox: (x, y, width, height) of detected face
            use_pid: Whether to use PID control
        
        Returns:
            (new_pan, new_tilt) target angles
        """
        x, y, w, h = face_bbox
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        # Check if face is within deadzone (center of image)
        image_center_x = self.config.image_width / 2
        image_center_y = self.config.image_height / 2
        
        x_error = face_center_x - image_center_x
        y_error = face_center_y - image_center_y
        
        if abs(x_error) < self.config.deadzone and abs(y_error) < self.config.deadzone:
            # Face is centered enough, no movement needed
            return self.current_pan, self.current_tilt
        
        # Convert to target angles
        target_pan_offset, target_tilt_offset = self.image_coords_to_angles(
            face_center_x, face_center_y, w, h
        )
        
        if use_pid:
            # Use PID for smooth movement
            pan_output = self.pan_pid.update(target_pan_offset)
            tilt_output = self.tilt_pid.update(target_tilt_offset)
            
            # Apply speed limits
            pan_output = max(-self.config.max_speed, 
                           min(self.config.max_speed, pan_output))
            tilt_output = max(-self.config.max_speed, 
                            min(self.config.max_speed, tilt_output))
            
            # Update target positions
            self.target_pan = self.current_pan + pan_output
            self.target_tilt = self.current_tilt + tilt_output
        else:
            # Direct positioning (no PID)
            self.target_pan = target_pan_offset
            self.target_tilt = target_tilt_offset
        
        # Apply smoothing
        self.smoothed_pan = (self.config.smoothing_factor * self.smoothed_pan + 
                            (1 - self.config.smoothing_factor) * self.target_pan)
        self.smoothed_tilt = (self.config.smoothing_factor * self.smoothed_tilt + 
                             (1 - self.config.smoothing_factor) * self.target_tilt)
        
        # Enforce limits
        self.smoothed_pan = max(self.config.pan_limits[0], 
                               min(self.config.pan_limits[1], self.smoothed_pan))
        self.smoothed_tilt = max(self.config.tilt_limits[0], 
                                min(self.config.tilt_limits[1], self.smoothed_tilt))
        
        return self.smoothed_pan, self.smoothed_tilt
    
    def move_to(self, pan: float, tilt: float):
        """Move gimbal to specified absolute angles."""
        # Enforce limits
        pan = max(self.config.pan_limits[0], min(self.config.pan_limits[1], pan))
        tilt = max(self.config.tilt_limits[0], min(self.config.tilt_limits[1], tilt))
        
        self._send_command(pan, tilt)
        self.current_pan = pan
        self.current_tilt = tilt
    
    def center_gimbal(self):
        """Return gimbal to center position."""
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.move_to(self.config.center_position[0], self.config.center_position[1])
    
    def stop(self):
        """Stop gimbal movement and cleanup."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        if self.serial_conn:
            self.serial_conn.close()
        logger.info("Gimbal controller stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current gimbal status."""
        return {
            "current_pan": self.current_pan,
            "current_tilt": self.current_tilt,
            "target_pan": self.target_pan,
            "target_tilt": self.target_tilt,
            "smoothed_pan": self.smoothed_pan,
            "smoothed_tilt": self.smoothed_tilt,
            "protocol": self.config.protocol.value,
            "connected": self.serial_conn is not None if self.config.protocol != GimbalProtocol.SIMULATION else True
        }


def create_gimbal_controller(config_path: str = None, 
                            simulation: bool = False) -> GimbalController:
    """
    Factory function to create a gimbal controller.
    
    Args:
        config_path: Path to configuration JSON file
        simulation: Force simulation mode
    
    Returns:
        Configured GimbalController instance
    """
    if config_path:
        config = GimbalConfig.from_file(config_path)
    else:
        config = GimbalConfig()
    
    if simulation:
        config.protocol = GimbalProtocol.SIMULATION
    
    return GimbalController(config)


if __name__ == "__main__":
    # Test/demo mode
    print("Testing Gimbal Controller...")
    
    # Create controller in simulation mode
    controller = create_gimbal_controller(simulation=True)
    
    # Simulate face detection
    test_faces = [
        {"bbox": (100, 100, 80, 80), "emotion": "happy", "confidence": 0.95},
        {"bbox": (400, 200, 60, 60), "emotion": "sad", "confidence": 0.85},
        {"bbox": (300, 150, 100, 100), "emotion": "surprise", "confidence": 0.92}
    ]
    
    # Select target
    target = controller.select_target_face(test_faces, priority="largest")
    if target:
        print(f"Selected target: {target}")
        
        # Update gimbal position
        pan, tilt = controller.update_target(target["bbox"])
        print(f"Moving to: pan={pan:.2f}, tilt={tilt:.2f}")
        
        # Get status
        status = controller.get_status()
        print(f"Status: {status}")
    
    # Test centering
    controller.center_gimbal()
    print("Centered gimbal")
    
    controller.stop()
    print("Test complete!")
