#!/usr/bin/env python3
"""
Insta360 One X2 + Jetson Orin Nano Emotion Recognition System

This script captures video from an Insta360 One X2 camera connected to
an NVIDIA Jetson Orin Nano and performs real-time emotion recognition.

Requirements:
    - OpenCV (cv2)
    - PyTorch with torchvision
    - NumPy
    - Transformers (for pre-trained models)

Usage:
    python3 emotion_recognition.py [--camera_id 0] [--rtsp_url <url>]
"""

import cv2
import numpy as np
import torch
import argparse
from collections import deque
import time


class EmotionRecognizer:
    """Real-time emotion recognition using pre-trained models."""
    
    # Emotion labels for FER (Facial Expression Recognition)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, model_type='fer2013', device=None):
        """
        Initialize the emotion recognizer.
        
        Args:
            model_type: Type of model to use ('fer2013', 'rafdb', or 'custom')
            device: CUDA device or 'cpu'
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load face detector (Haar Cascade - lightweight for edge devices)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # For better accuracy, you can use dlib or MTCNN
        # self.face_detector = dlib.get_frontal_face_detector()
        
        # Simple CNN-based emotion classifier (you can replace with pre-trained model)
        self.emotion_model = self._build_emotion_model()
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Smoothing for stable predictions
        self.prediction_history = deque(maxlen=10)
        
    def _build_emotion_model(self):
        """
        Build a simple emotion classification model.
        In production, load a pre-trained model instead.
        """
        import torch.nn as nn
        
        class SimpleEmotionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128 * 6 * 6, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 7)  # 7 emotions
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        model = SimpleEmotionNet()
        
        # Try to load pre-trained weights if available
        try:
            # Uncomment when you have trained weights
            # model.load_state_dict(torch.load('emotion_model.pth', map_location=self.device))
            print("Using untrained model - consider training on FER2013 dataset")
        except FileNotFoundError:
            pass
        
        return model
    
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion model."""
        import torch.nn.functional as F
        
        # Resize to model input size
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        
        # Normalize
        mean = 0.485
        std = 0.229
        face_img = (face_img - mean) / std
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_img):
        """Predict emotion from a face image."""
        face_tensor = self.preprocess_face(face_img)
        
        with torch.no_grad():
            outputs = self.emotion_model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion_idx = predicted.item()
        conf = confidence.item()
        
        # Add to history for smoothing
        self.prediction_history.append((emotion_idx, conf))
        
        # Get most common recent prediction
        recent_emotions = [p[0] for p in self.prediction_history]
        smoothed_emotion = max(set(recent_emotions), key=recent_emotions.count)
        
        return self.EMOTIONS[smoothed_emotion], conf
    
    def process_frame(self, frame):
        """
        Process a single frame for face detection and emotion recognition.
        
        Returns:
            frame: Annotated frame with bounding boxes and emotion labels
            results: List of (face_box, emotion, confidence) tuples
        """
        faces, gray = self.detect_faces(frame)
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_roi)
            
            # Store result
            results.append(((x, y, w, h), emotion, confidence))
            
            # Draw bounding box
            color = self._get_emotion_color(emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, results
    
    def _get_emotion_color(self, emotion):
        """Get BGR color for emotion visualization."""
        colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Surprise': (255, 255, 0), # Cyan
            'Fear': (128, 0, 128),     # Purple
            'Disgust': (0, 128, 128),  # Teal
            'Neutral': (255, 255, 255) # White
        }
        return colors.get(emotion, (255, 255, 255))


class CameraCapture:
    """Handle camera capture from Insta360 One X2."""
    
    def __init__(self, source=0, rtsp_url=None, width=1280, height=720):
        """
        Initialize camera capture.
        
        Args:
            source: Video device ID (e.g., 0 for /dev/video0)
            rtsp_url: RTSP stream URL (alternative to direct USB)
            width: Desired capture width
            height: Desired capture height
        """
        self.source = source
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = None
        
    def open(self):
        """Open the camera stream."""
        if self.rtsp_url:
            print(f"Opening RTSP stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            print(f"Opening camera device: /dev/video{self.source}")
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera source")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set FPS (if supported)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Print actual settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        return True
    
    def read(self):
        """Read a frame from the camera."""
        if self.cap is None:
            return None, None
        return self.cap.read()
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main():
    """Main entry point for emotion recognition system."""
    parser = argparse.ArgumentParser(description='Insta360 One X2 Emotion Recognition')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--rtsp_url', type=str, default=None,
                       help='RTSP stream URL (alternative to USB)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Capture width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Capture height (default: 720)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (headless mode)')
    
    args = parser.parse_args()
    
    # Initialize components
    print("=" * 60)
    print("Insta360 One X2 Emotion Recognition System")
    print("=" * 60)
    
    try:
        # Initialize camera
        camera = CameraCapture(
            source=args.camera_id,
            rtsp_url=args.rtsp_url,
            width=args.width,
            height=args.height
        )
        camera.open()
        
        # Initialize emotion recognizer
        recognizer = EmotionRecognizer()
        
        # Optional: Video writer for recording
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, 20.0, 
                                   (args.width, args.height))
        
        print("\nStarting emotion recognition... Press 'q' to quit")
        print("-" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for emotion recognition
            processed_frame, results = recognizer.process_frame(frame)
            
            # Display results info
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f}, Faces detected: {len(results)}")
            
            # Write to output file if specified
            if writer:
                writer.write(processed_frame)
            
            # Display frame (unless headless)
            if not args.no_display:
                cv2.imshow('Emotion Recognition - Insta360 One X2', processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        elapsed = time.time() - start_time
        print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        camera.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")


if __name__ == '__main__':
    main()
