# Emotion Recognition Model Training Guide

Complete guide for training custom emotion recognition models optimized for NVIDIA Jetson Orin Nano deployment.

## Overview

This training pipeline supports:
- **FER2013 Dataset** - Standard facial expression recognition dataset (35,887 images)
- **Custom Datasets** - Your own collected data from Insta360 camera
- **Transfer Learning** - Pretrained MobileNetV3 for better accuracy
- **TensorRT Deployment** - ONNX export for optimal Jetson performance

## Quick Start

### 1. Install Dependencies

```bash
./install_training.sh
```

### 2. Download FER2013 Dataset

Download from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Place `fer2013.csv` in `./data/fer2013/`

### 3. Train Model

```bash
# Train custom lightweight model
python3 train_emotion.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --epochs 50 \
    --batch-size 64 \
    --output-dir ./checkpoints \
    --export-onnx

# OR train with pretrained MobileNetV3 (recommended for better accuracy)
python3 train_emotion.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --use-pretrained \
    --epochs 30 \
    --output-dir ./checkpoints_mobilenet \
    --export-onnx
```

## Dataset Options

### Option 1: FER2013 (Recommended for starting)

The FER2013 dataset contains 48x48 grayscale face images labeled with 7 emotions:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

**Structure:**
```
data/
└── fer2013/
    └── fer2013.csv
```

### Option 2: Custom Dataset

Collect your own data using the Insta360 camera in your specific environment for better real-world performance.

**Required Structure:**
```
data/
└── custom_emotions/
    ├── train/
    │   ├── 0_angry/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── 1_disgust/
    │   ├── 2_fear/
    │   ├── 3_happy/
    │   ├── 4_sad/
    │   ├── 5_surprise/
    │   └── 6_neutral/
    └── val/
        ├── 0_angry/
        └── ...
```

**Training Command:**
```bash
python3 train_emotion.py \
    --dataset custom \
    --data-dir ./data/custom_emotions \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir ./checkpoints_custom \
    --export-onnx
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | fer2013 | Dataset type: 'fer2013' or 'custom' |
| `--data-dir` | (required) | Path to dataset directory |
| `--train-path` | auto | Path to training CSV (FER2013 only) |
| `--val-path` | auto | Path to validation CSV (FER2013 only) |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size (reduce if OOM) |
| `--lr` | 0.001 | Learning rate |
| `--use-pretrained` | False | Use pretrained MobileNetV3 |
| `--output-dir` | ./checkpoints | Directory to save checkpoints |
| `--export-onnx` | False | Export model to ONNX format |

## Model Architecture

### EmotionNet (Custom Lightweight CNN)
- Optimized for Jetson Orin Nano
- ~500K parameters
- 7 convolutional blocks with batch normalization
- Adaptive average pooling
- Two fully connected layers with dropout

### MobileNetV3 (Pretrained Option)
- Transfer learning from ImageNet
- Better accuracy, slightly larger
- Recommended for production use

## Training Output

After training, you'll get:
- `best_model.pth` - Best performing model checkpoint
- `checkpoint_epoch_X.pth` - Checkpoints every 10 epochs
- `training_history.png` - Loss and accuracy curves
- `emotion_model.onnx` - ONNX model for TensorRT conversion (if --export-onnx)

## Deploy on Jetson Orin Nano

### Step 1: Transfer Model to Jetson

```bash
scp ./checkpoints/best_model.pth jetson@<ip>:/home/jetson/models/
scp ./checkpoints/emotion_model.onnx jetson@<ip>:/home/jetson/models/
```

### Step 2: Convert to TensorRT Engine (on Jetson)

```bash
# Install TensorRT if not already installed
sudo apt-get install tensorrt

# Convert ONNX to TensorRT engine with FP16 precision
trtexec --onnx=/home/jetson/models/emotion_model.onnx \
        --saveEngine=/home/jetson/models/emotion_model.engine \
        --fp16 \
        --minShapes=input:1x3x48x48 \
        --optShapes=input:4x3x48x48 \
        --maxShapes=input:8x3x48x48
```

### Step 3: Update Inference Script

Modify `emotion_recognition.py` to load your trained model:

```python
# Load trained model
model = EmotionNet(num_classes=7)
checkpoint = torch.load('./models/best_model.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda().eval()

# OR use TensorRT engine
import pyCUDA
from pyCUDA import infer

# Load TensorRT engine
with open('./models/emotion_model.engine', 'rb') as f:
    engine_data = f.read()
runtime = infer.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(engine_data)
```

## Tips for Better Results

### 1. Data Augmentation
Already included in training:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)

### 2. Handle Class Imbalance
If your dataset has imbalanced classes:

```python
# Add to create_model() in train_emotion.py
class_weights = torch.tensor([1.5, 1.5, 1.2, 0.8, 1.3, 1.0, 0.9]).to(device)
self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 3. Hyperparameter Tuning
- **Learning Rate**: Try 0.001, 0.0005, 0.0001
- **Batch Size**: 32, 64, 128 (depending on GPU memory)
- **Dropout**: 0.3, 0.4, 0.5
- **Epochs**: 30-100 (use early stopping)

### 4. Transfer Learning
Always start with pretrained MobileNetV3 for better results:
```bash
python3 train_emotion.py --use-pretrained --epochs 30
```

### 5. Custom Data Collection
For best results in your specific environment:
1. Record video with Insta360 camera
2. Extract frames using OpenCV
3. Detect faces and crop
4. Label emotions manually or with existing model
5. Train on this custom dataset

## Expected Performance

| Model | Parameters | FER2013 Accuracy | Jetson FPS |
|-------|-----------|------------------|------------|
| EmotionNet | ~500K | 65-72% | 60-100 |
| MobileNetV3 | ~2.5M | 70-78% | 40-60 |
| MobileNetV3 (TensorRT) | ~2.5M | 70-78% | 80-120 |

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python3 train_emotion.py --batch-size 32
```

### Slow Training on Jetson
- Enable maximum power mode: `sudo nvpmodel -m 0`
- Set GPU to max frequency: `sudo jetson_clocks`
- Use mixed precision (already enabled in script)

### Poor Accuracy
- Increase training epochs
- Use pretrained model (--use-pretrained)
- Collect more training data
- Verify data preprocessing matches inference

### ONNX Export Fails
Ensure you're using compatible PyTorch version:
```bash
pip3 install torch==2.0.0 torchvision==0.15.0
```

## Advanced: INT8 Quantization

For maximum performance on Jetson:

```bash
# Create calibration dataset (500 representative images)
mkdir ./calibration_data
# Add images to calibration_data/

# Convert to INT8 TensorRT engine
trtexec --onnx=emotion_model.onnx \
        --saveEngine=emotion_model_int8.engine \
        --int8 \
        --calib=./calibration_data \
        --fp16
```

This can provide 2-3x speedup with minimal accuracy loss.

## Next Steps

1. Train initial model on FER2013
2. Test on your Insta360 camera feed
3. Collect misclassified frames
4. Retrain with augmented dataset
5. Deploy with TensorRT for production

For questions or issues, check:
- PyTorch documentation: https://pytorch.org/docs/
- NVIDIA TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
- FER2013 paper: https://arxiv.org/abs/1302.3181
