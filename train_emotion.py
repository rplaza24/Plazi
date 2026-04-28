#!/usr/bin/env python3
"""
Train emotion recognition model optimized for NVIDIA Jetson Orin Nano
Supports FER2013, AffectNet, or custom datasets
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.cuda import amp
import torch.nn.functional as F

# Check if running on Jetson
IS_JETSON = os.path.exists('/etc/nv_tegra_release')


class FER2013Dataset(Dataset):
    """Custom dataset loader for FER2013 CSV format"""
    
    def __init__(self, csv_path, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Parse pixel string to numpy array
        pixels = np.array(list(map(int, self.data.iloc[idx]['pixels'].split())), dtype=np.float32)
        image = pixels.reshape(48, 48)
        
        # Convert to RGB (3 channels) for model compatibility
        image = np.stack([image] * 3, axis=-1)
        
        label = self.data.iloc[idx]['emotion']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EmotionNet(nn.Module):
    """
    Lightweight CNN optimized for Jetson Orin Nano
    Based on MobileNetV3 architecture principles
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(EmotionNet, self).__init__()
        
        # Feature extraction - simplified MobileNet-style
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 5
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 6
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 7
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EmotionTrainer:
    """Training pipeline for emotion recognition models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 7
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        print(f"Using device: {self.device}")
        if IS_JETSON:
            print("✓ Running on NVIDIA Jetson")
        else:
            print("ℹ Running on desktop/server (model will be compatible with Jetson)")
        
        # Data transformations
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_data(self):
        """Load dataset based on provided path"""
        if self.args.dataset == 'fer2013':
            train_data = FER2013Dataset(self.args.train_path, transform=self.train_transform)
            val_data = FER2013Dataset(self.args.val_path, transform=self.val_transform)
        else:
            # ImageFolder for custom datasets
            train_data = ImageFolder(os.path.join(self.args.data_dir, 'train'), 
                                   transform=self.train_transform)
            val_data = ImageFolder(os.path.join(self.args.data_dir, 'val'), 
                                 transform=self.val_transform)
        
        self.train_loader = DataLoader(train_data, batch_size=self.args.batch_size, 
                                      shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_data, batch_size=self.args.batch_size, 
                                    shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
    
    def create_model(self):
        """Create and initialize the model"""
        if self.args.use_pretrained:
            # Use pretrained MobileNetV3 and modify classifier
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            num_features = base_model.classifier[3].in_features
            base_model.classifier[3] = nn.Linear(num_features, self.num_classes)
            self.model = base_model
        else:
            self.model = EmotionNet(num_classes=self.num_classes)
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.args.lr, 
                                    weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs
        )
        
        # Mixed precision scaler for faster training on Jetson
        self.scaler = amp.GradScaler()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch}/{self.args.epochs}] | '
                      f'Batch [{i}/{len(self.train_loader)}] | '
                      f'Loss: {running_loss/(i+1):.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self):
        """Full training loop"""
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print("\nStarting training...")
        print("="*60)
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"\nResults:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(f"{self.args.output_dir}/best_model.pth", epoch, val_acc)
                print(f"✓ New best model saved! Accuracy: {val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_model(f"{self.args.output_dir}/checkpoint_epoch_{epoch}.pth", 
                              epoch, val_acc)
        
        print("\n" + "="*60)
        print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
        
        # Plot training curves
        self.plot_history(history)
        
        # Export to ONNX for TensorRT deployment
        if self.args.export_onnx:
            self.export_onnx()
    
    def save_model(self, path, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'args': vars(self.args)
        }
        torch.save(checkpoint, path)
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.args.output_dir}/training_history.png")
        print(f"✓ Training history plot saved to {self.args.output_dir}/training_history.png")
    
    def export_onnx(self):
        """Export model to ONNX format for TensorRT deployment"""
        self.model.eval()
        dummy_input = torch.randn(1, 3, 48, 48).to(self.device)
        
        onnx_path = f"{self.args.output_dir}/emotion_model.onnx"
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ Model exported to ONNX: {onnx_path}")
        print("  Next step: Convert to TensorRT engine for optimal Jetson performance")
        print("  Command: trtexec --onnx=emotion_model.onnx --saveEngine=emotion_model.engine")


def prepare_fer2013_data(data_dir):
    """
    Prepare FER2013 dataset by splitting into train/val
    Download from: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    """
    import pandas as pd
    
    print("Preparing FER2013 dataset...")
    
    # Load CSV
    df = pd.read_csv(os.path.join(data_dir, 'fer2013.csv'))
    
    # Create directory structure
    for split in ['train', 'val']:
        for emotion in range(7):
            os.makedirs(os.path.join(data_dir, split, str(emotion)), exist_ok=True)
    
    # Split data (90% train, 10% val)
    train_df = df[df['Usage'] == 'Training'].sample(frac=0.9, random_state=42)
    val_df = df[df['Usage'] == 'Training'].drop(train_df.index)
    
    # Save split CSVs
    train_df.to_csv(os.path.join(data_dir, 'fer2013_train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'fer2013_val.csv'), index=False)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print("✓ FER2013 dataset prepared!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model for Jetson')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='fer2013', 
                       choices=['fer2013', 'custom'],
                       help='Dataset type')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--train-path', type=str,
                       help='Path to training CSV (for FER2013)')
    parser.add_argument('--val-path', type=str,
                       help='Path to validation CSV (for FER2013)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pretrained MobileNetV3')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export model to ONNX format')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset if needed
    if args.dataset == 'fer2013' and not args.train_path:
        prepare_fer2013_data(args.data_dir)
        args.train_path = os.path.join(args.data_dir, 'fer2013_train.csv')
        args.val_path = os.path.join(args.data_dir, 'fer2013_val.csv')
    
    # Initialize trainer
    trainer = EmotionTrainer(args)
    
    # Load data
    trainer.load_data()
    
    # Create model
    trainer.create_model()
    
    # Train
    trainer.train()
