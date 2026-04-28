#!/usr/bin/env python3
"""
Emotion-Driven Gimbal Tracker
Integrates emotion recognition with gimbal control for automatic face tracking.
Targets specific emotions and keeps them centered in the camera frame.
"""

import cv2
import numpy as np
import argparse
import logging
import time
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import our modules
from gimbal_controller import GimbalController, create_gimbal_controller, GimbalConfig

# Try to import emotion recognition components
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using mock emotion detection.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """Lightweight emotion detector for real-time inference."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_path: str = None, use_pretrained: bool = True,
                 device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transform = None
        
        if TORCH_AVAILABLE:
            self._load_model(model_path, use_pretrained)
        else:
            logger.warning("Using mock emotion detector (PyTorch not available)")
    
    def _load_model(self, model_path: str, use_pretrained: bool):
        """Load trained emotion recognition model."""
        try:
            # Use MobileNetV3 for good accuracy/speed tradeoff on Jetson
            if use_pretrained:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                self.model = models.mobilenet_v3_small(weights=weights)
                
                # Replace classifier for 7 emotions
                num_features = self.model.classifier[0].in_features
                self.model.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, len(self.EMOTIONS))
                )
            else:
                # Load custom trained model
                if model_path and Path(model_path).exists():
                    self.model = self._load_custom_model(model_path)
                else:
                    logger.warning("No model found, using random initialization")
                    self.model = self._create_default_model()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"Emotion detector loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using mock detector.")
            self.model = None
    
    def _create_default_model(self) -> nn.Module:
        """Create a simple default model."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.EMOTIONS))
        )
        return model
    
    def _load_custom_model(self, model_path: str) -> nn.Module:
        """Load custom trained model from checkpoint."""
        # This should match your training architecture
        checkpoint = torch.load(model_path, map_location=self.device)
        model = self._create_default_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def detect_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Detect emotion from a face image.
        
        Args:
            face_image: BGR image of face (OpenCV format)
        
        Returns:
            (emotion_label, confidence_score)
        """
        if self.model is None or not TORCH_AVAILABLE:
            # Mock detection for testing
            return self._mock_detect(face_image)
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
            
            emotion = self.EMOTIONS[predicted.item()]
            conf = confidence.item()
            
            return emotion, conf
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return "neutral", 0.5
    
    def _mock_detect(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Mock emotion detection for testing without model."""
        import random
        emotion = random.choice(self.EMOTIONS)
        confidence = random.uniform(0.6, 0.95)
        return emotion, confidence


class EmotionGimbalTracker:
    """Main tracker combining emotion detection and gimbal control."""
    
    def __init__(self, 
                 camera_source: str = "/dev/video0",
                 gimbal_port: str = "/dev/ttyTHS1",
                 config_path: str = None,
                 target_emotion: str = None,
                 confidence_threshold: float = 0.6,
                 priority: str = "largest",
                 use_pretrained: bool = True,
                 model_path: str = None,
                 headless: bool = False,
                 simulation: bool = False):
        """
        Initialize the emotion-driven gimbal tracker.
        
        Args:
            camera_source: Video source (device path or RTSP URL)
            gimbal_port: Serial port for gimbal communication
            config_path: Path to gimbal configuration JSON
            target_emotion: Specific emotion to track (None for all)
            confidence_threshold: Minimum confidence for detection
            priority: Target selection priority ("largest", "closest_to_center")
            use_pretrained: Use pretrained model or custom
            model_path: Path to custom trained model
            headless: Run without display (for embedded deployment)
            simulation: Run in simulation mode (no hardware)
        """
        self.camera_source = camera_source
        self.target_emotion = target_emotion.lower() if target_emotion else None
        self.confidence_threshold = confidence_threshold
        self.priority = priority
        self.headless = headless
        
        # Initialize gimbal controller
        if config_path:
            self.gimbal = create_gimbal_controller(config_path, simulation=simulation)
        else:
            config = GimbalConfig(port=gimbal_port)
            if simulation:
                config.protocol = config.protocol.__class__.SIMULATION
            self.gimbal = GimbalController(config)
        
        # Initialize emotion detector
        self.emotion_detector = EmotionDetector(
            model_path=model_path,
            use_pretrained=use_pretrained
        )
        
        # OpenCV setup
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps_history = []
        
        # Tracking state
        self.current_target = None
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # seconds before giving up on lost target
        
        # Colors for visualization
        self.colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 255, 255),  # Cyan
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 255, 0), # Yellow
            'neutral': (255, 255, 255) # White
        }
    
    def open_camera(self) -> bool:
        """Open camera connection."""
        try:
            # Try different backends for Jetson
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_source, backend)
                if self.cap.isOpened():
                    # Set common properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    logger.info(f"Camera opened successfully: {self.camera_source}")
                    logger.info(f"Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                               f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                    return True
                self.cap.release()
            
            logger.error("Failed to open camera with any backend")
            return False
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame using OpenCV Haar cascades.
        
        Returns:
            List of face dictionaries with bbox, emotion, confidence
        """
        # Load cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process detections
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotion, confidence = self.emotion_detector.detect_emotion(face_roi)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence
            })
        
        return results
    
    def draw_detections(self, frame: np.ndarray, faces: List[Dict], 
                       selected_face: Optional[Dict] = None):
        """Draw face detections and emotions on frame."""
        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            emotion = face['emotion']
            conf = face['confidence']
            
            # Choose color based on emotion
            color = self.colors.get(emotion, (255, 255, 255))
            
            # Highlight selected target
            if selected_face and face == selected_face:
                thickness = 3
                # Draw outer rectangle
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 255), thickness)
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw label background
            label = f"{emotion}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw gimbal status
        status = self.gimbal.get_status()
        status_text = [
            f"PAN: {status['current_pan']:.1f}",
            f"TILT: {status['current_tilt']:.1f}",
            f"FPS: {self._get_fps():.1f}",
            f"Target: {self.target_emotion or 'ALL'}"
        ]
        
        y_offset = 30
        for text in status_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    def _get_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_history.append(current_time)
        
        # Keep only last second of frames
        self.fps_history = [t for t in self.fps_history if current_time - t < 1.0]
        
        return len(self.fps_history)
    
    def run(self):
        """Main tracking loop."""
        if not self.open_camera():
            logger.error("Failed to open camera. Exiting.")
            return
        
        self.running = True
        logger.info("Starting emotion-driven gimbal tracking...")
        logger.info(f"Target emotion: {self.target_emotion or 'ALL'}")
        logger.info(f"Priority: {self.priority}")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Select target face
                target_emotions = [self.target_emotion] if self.target_emotion else None
                selected_face = self.gimbal.select_target_face(
                    faces,
                    priority=self.priority,
                    target_emotions=target_emotions,
                    confidence_threshold=self.confidence_threshold
                )
                
                # Update gimbal position if target found
                if selected_face:
                    self.current_target = selected_face
                    self.last_detection_time = time.time()
                    
                    pan, tilt = self.gimbal.update_target(selected_face['bbox'])
                    self.gimbal.move_to(pan, tilt)
                    
                    logger.debug(f"Tracking {selected_face['emotion']} face "
                               f"at ({selected_face['bbox'][0]}, {selected_face['bbox'][1]})")
                else:
                    # Check if we've lost the target
                    if self.current_target and \
                       (time.time() - self.last_detection_time) > self.detection_timeout:
                        logger.info("Target lost, centering gimbal")
                        self.gimbal.center_gimbal()
                        self.current_target = None
                
                # Draw visualizations
                if not self.headless:
                    self.draw_detections(frame, faces, selected_face)
                    cv2.imshow('Emotion Tracker - Press Q to quit', frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # Q or ESC
                        logger.info("Quit requested")
                        break
                    elif key == ord('c'):  # C to center
                        logger.info("Centering gimbal manually")
                        self.gimbal.center_gimbal()
                    elif key == ord('r'):  # R to recalibrate
                        logger.info("Recalibrating...")
                        self.gimbal.center_gimbal()
                        time.sleep(1)
                
                # Calculate loop timing
                loop_time = time.time() - start_time
                target_fps = 30
                if loop_time < (1.0 / target_fps):
                    time.sleep((1.0 / target_fps) - loop_time)
                
                # Log FPS periodically
                if self.frame_count % 60 == 0:
                    fps = self._get_fps()
                    logger.info(f"Running at {fps:.1f} FPS")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Tracking error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop tracking and cleanup resources."""
        logger.info("Stopping tracker...")
        self.running = False
        
        # Center gimbal before stopping
        if hasattr(self, 'gimbal'):
            self.gimbal.center_gimbal()
            self.gimbal.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        if not self.headless:
            cv2.destroyAllWindows()
        
        logger.info("Tracker stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Emotion-driven gimbal tracker for Insta360 One X2'
    )
    
    parser.add_argument('--camera-source', type=str, default='/dev/video0',
                       help='Camera device path or RTSP URL')
    parser.add_argument('--gimbal-port', type=str, default='/dev/ttyTHS1',
                       help='Serial port for gimbal')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to gimbal configuration JSON')
    parser.add_argument('--target-emotion', type=str, default=None,
                       choices=['angry', 'disgust', 'fear', 'happy', 'sad', 
                               'surprise', 'neutral'],
                       help='Specific emotion to track')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence for detection')
    parser.add_argument('--priority', type=str, default='largest',
                       choices=['largest', 'closest_to_center'],
                       help='Target selection priority')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to custom trained emotion model')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                       help='Use pretrained model (default: True)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (embedded mode)')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (no hardware)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure file logging if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Create and run tracker
    tracker = EmotionGimbalTracker(
        camera_source=args.camera_source,
        gimbal_port=args.gimbal_port,
        config_path=args.config,
        target_emotion=args.target_emotion,
        confidence_threshold=args.confidence_threshold,
        priority=args.priority,
        use_pretrained=args.use_pretrained,
        model_path=args.model_path,
        headless=args.headless,
        simulation=args.simulation
    )
    
    tracker.run()


if __name__ == "__main__":
    main()
