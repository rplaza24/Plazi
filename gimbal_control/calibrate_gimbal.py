#!/usr/bin/env python3
"""
Gimbal Calibration Utility
Helps calibrate the gimbal by mapping image coordinates to pan/tilt angles.
Includes PID tuning and configuration testing.
"""

import cv2
import numpy as np
import argparse
import logging
import time
import json
from pathlib import Path

from gimbal_controller import GimbalController, create_gimbal_controller, GimbalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GimbalCalibrator:
    """Interactive gimbal calibration utility."""
    
    def __init__(self, config_path: str = None, simulation: bool = False):
        if config_path:
            self.config = GimbalConfig.from_file(config_path)
        else:
            self.config = GimbalConfig()
        
        if simulation:
            from gimbal_controller import GimbalProtocol
            self.config.protocol = GimbalProtocol.SIMULATION
        
        self.controller = GimbalController(self.config)
        self.calibration_points = []
        self.current_step = 0
        self.total_steps = 5
    
    def manual_control(self):
        """Manual gimbal control for testing."""
        print("\n=== Manual Gimbal Control ===")
        print("Controls:")
        print("  Arrow keys: Pan/Tilt movement")
        print("  Space: Center gimbal")
        print("  S: Save current position")
        print("  Q: Quit")
        print("=" * 40)
        
        # Create a blank window for key events
        cv2.namedWindow('Manual Control')
        
        pan, tilt = 0, 0
        step_size = 5.0
        
        print(f"Starting position: pan={pan:.1f}, tilt={tilt:.1f}")
        
        try:
            while True:
                # Display current status
                status = self.controller.get_status()
                print(f"\rCurrent: PAN={status['current_pan']:.1f} TILT={status['current_tilt']:.1f}", 
                      end='', flush=True)
                
                key = cv2.waitKey(10) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    self.controller.center_gimbal()
                    print("\nCentering gimbal...")
                elif key == 83:  # Right arrow
                    pan += step_size
                    self.controller.move_to(pan, tilt)
                elif key == 81:  # Left arrow
                    pan -= step_size
                    self.controller.move_to(pan, tilt)
                elif key == 82:  # Up arrow
                    tilt += step_size
                    self.controller.move_to(pan, tilt)
                elif key == 84:  # Down arrow
                    tilt -= step_size
                    self.controller.move_to(pan, tilt)
                elif key == ord('s'):
                    self.calibration_points.append((pan, tilt))
                    print(f"\nSaved position {len(self.calibration_points)}: ({pan:.1f}, {tilt:.1f})")
                elif key == ord('+') or key == ord('='):
                    step_size = min(step_size + 1, 20)
                    print(f"\nStep size: {step_size:.1f}")
                elif key == ord('-') or key == ord('_'):
                    step_size = max(step_size - 1, 1)
                    print(f"\nStep size: {step_size:.1f}")
        
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.controller.stop()
    
    def auto_calibrate(self):
        """Automatic calibration routine."""
        print("\n=== Automatic Calibration ===")
        print("This will move the gimbal to predefined positions.")
        print("Make sure the gimbal has clear range of motion!")
        input("Press Enter to start...")
        
        # Define calibration positions
        positions = [
            (0, 0, "Center"),
            (-45, 0, "Left"),
            (45, 0, "Right"),
            (0, -20, "Up"),
            (0, 20, "Down"),
            (-60, -15, "Top-Left"),
            (60, 15, "Bottom-Right")
        ]
        
        for i, (pan, tilt, name) in enumerate(positions):
            print(f"\nStep {i+1}/{len(positions)}: Moving to {name} ({pan}, {tilt})")
            self.controller.move_to(pan, tilt)
            time.sleep(1.5)
            
            status = self.controller.get_status()
            print(f"  Actual position: PAN={status['current_pan']:.1f}, "
                  f"TILT={status['current_tilt']:.1f}")
            
            if i < len(positions) - 1:
                input("  Press Enter to continue...")
        
        print("\nCalibration complete!")
        self.controller.center_gimbal()
    
    def tune_pid(self):
        """Interactive PID tuning."""
        print("\n=== PID Tuning ===")
        print("This will help you find optimal PID parameters.")
        print("\nCurrent PID values:")
        print(f"  Pan:  Kp={self.config.pid_pan.kp}, Ki={self.config.pid_pan.ki}, Kd={self.config.pid_pan.kd}")
        print(f"  Tilt: Kp={self.config.pid_tilt.kp}, Ki={self.config.pid_tilt.ki}, Kd={self.config.pid_tilt.kd}")
        
        print("\nTuning procedure:")
        print("1. Set Ki=0, Kd=0")
        print("2. Increase Kp until oscillation, then reduce by 50%")
        print("3. Increase Kd to reduce overshoot")
        print("4. Add Ki to eliminate steady-state error")
        
        # Test positions
        test_angles = [(30, 0), (-30, 0), (0, 15), (0, -15)]
        
        kp = float(input("\nEnter Kp (start with 0.5): ") or "0.5")
        ki = float(input("Enter Ki (start with 0.0): ") or "0.0")
        kd = float(input("Enter Kd (start with 0.1): ") or "0.1")
        
        # Update PID config
        from gimbal_controller import PIDConfig
        self.config.pid_pan = PIDConfig(kp, ki, kd)
        self.config.pid_tilt = PIDConfig(kp, ki, kd)
        
        # Recreate controller with new PID
        self.controller.pan_pid = self.controller.PIDController(self.config.pid_pan)
        self.controller.tilt_pid = self.controller.PIDController(self.config.pid_tilt)
        
        print(f"\nTesting with Kp={kp}, Ki={ki}, Kd={kd}")
        
        for pan, tilt in test_angles:
            print(f"\nMoving to ({pan}, {tilt})...")
            self.controller.update_target((320 + pan*5, 240 + tilt*5, 50, 50))
            self.controller.move_to(pan, tilt)
            time.sleep(2)
            
            status = self.controller.get_status()
            print(f"  Reached: PAN={status['smoothed_pan']:.1f}, TILT={status['smoothed_tilt']:.1f}")
        
        save = input("\nSave these PID values? (y/n): ").lower()
        if save == 'y':
            self.config.to_file('gimbal_config_tuned.json')
            print("Saved to gimbal_config_tuned.json")
    
    def save_configuration(self, filepath: str = None):
        """Save current configuration."""
        if filepath is None:
            filepath = f"gimbal_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        self.config.to_file(filepath)
        print(f"Configuration saved to {filepath}")
    
    def run_calibration_wizard(self):
        """Run full calibration wizard."""
        print("\n" + "=" * 60)
        print("   GIMBAL CALIBRATION WIZARD")
        print("=" * 60)
        print("\nThis wizard will help you calibrate your gimbal system.")
        print("Please ensure:")
        print("  ✓ Gimbal is properly connected")
        print("  ✓ Camera is mounted and working")
        print("  ✓ Clear space for gimbal movement")
        print()
        
        while True:
            print("\nSelect an option:")
            print("1. Manual Control (test movement)")
            print("2. Auto Calibration (predefined positions)")
            print("3. PID Tuning (optimize response)")
            print("4. Save Configuration")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.manual_control()
            elif choice == '2':
                self.auto_calibrate()
            elif choice == '3':
                self.tune_pid()
            elif choice == '4':
                filename = input("Enter filename (or press Enter for auto-generated): ").strip()
                self.save_configuration(filename if filename else None)
            elif choice == '5':
                print("Exiting calibration wizard.")
                self.controller.stop()
                break
            else:
                print("Invalid choice. Please enter 1-5.")


def parse_args():
    parser = argparse.ArgumentParser(description='Gimbal Calibration Utility')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='wizard',
                       choices=['wizard', 'manual', 'auto', 'pid'],
                       help='Calibration mode')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode')
    return parser.parse_args()


def main():
    args = parse_args()
    
    calibrator = GimbalCalibrator(
        config_path=args.config,
        simulation=args.simulation
    )
    
    if args.mode == 'wizard':
        calibrator.run_calibration_wizard()
    elif args.mode == 'manual':
        calibrator.manual_control()
    elif args.mode == 'auto':
        calibrator.auto_calibrate()
    elif args.mode == 'pid':
        calibrator.tune_pid()


if __name__ == "__main__":
    main()
