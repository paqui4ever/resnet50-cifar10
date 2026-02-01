<div align="center">

# ResNet50 CIFAR-10 Classification

**Deep Residual Learning for CIFAR-10 Image Classification (Trained from Scratch)**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DirectML](https://img.shields.io/badge/DirectML-0078D4?style=flat&logo=microsoft&logoColor=white)](https://github.com/microsoft/DirectML)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)

</div>

---

## 📋 Overview

This repository implements a **ResNet50 model trained from scratch** for CIFAR-10 image classification, leveraging DirectML for AMD GPU acceleration. The model is trained without pretrained weights, achieving strong performance through careful data augmentation, learning rate scheduling, and training optimizations. With all this, the model achieves a **test accuracy of ~86%**.

---

## 🗂️ Project Structure

```
resnet50-cifar10-main/
├── 📁 data/
│   ├── __init__.py           # Package initialization
│   └── transforms.py         # Data transformation pipelines
├── 📁 notebooks/
│   └── cifar_10.ipynb        # Initial analysis & experiments
├── 📁 runs/                  # TensorBoard logs 
├── 🐍 train.py               # Model training script with TensorBoard
├── 🐍 evaluate.py            # Evaluation & visualization
├── 📄 requirements.txt       # Python dependencies
└── 📄 README.md              
```

---

## 🔧 Setup

Simply run:

```bash
# Clone the repository
git clone <repository-url>
cd resnet50-cifar10

# Install all dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Training

Train the ResNet50 model from scratch on CIFAR-10:

```bash
python train.py
```

**Training Process:**
- Downloads CIFAR-10 dataset (auto-downloaded on first run)
- Splits data: 40k train / 10k validation / 10k test
- Trains for 100 epochs with cosine annealing LR schedule
- Logs metrics to TensorBoard in `runs/` directory
- Saves model checkpoint as `resnet50_cifar10.pth`

### Monitoring Training

While training is running (or after it completes), visualize metrics with TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser to view:
- Training loss per step
- Train/validation/test accuracy per epoch
- Learning rate schedule

### Evaluation

Evaluate the trained model and visualize predictions:

```bash
python evaluate.py
```

**Evaluation Features:**
- Computes test set accuracy
- Visualizes sample predictions with ground truth labels
- Shows prediction correctness indicators

---

## 🏗️ Model Architecture

### Network Details

| Component | Specification |
|-----------|--------------|
| **Base Architecture** | ResNet50 (trained from scratch) |
| **Weights** | Randomly initialized |
| **Input Size** | 32×32×3 (CIFAR-10 images) |
| **Output Classes** | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | SGD (momentum=0.9, weight_decay=5e-4) |
| **Learning Rate** | 0.1 (initial) |
| **LR Scheduler** | CosineAnnealingLR |
| **Batch Size** | 128 |
| **Epochs** | 100 |
| **Loss Function** | CrossEntropyLoss |

### Data Pipeline

**Training Augmentation:**
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- RandAugment (num_ops=3, magnitude=9)
- Normalization (ImageNet mean/std)

**Test/Validation:**
- Normalization only (no augmentation)

---

## 🎓 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~86% |
| **Training Time** | ~100 epochs |

---

## 📚 References

- He et al., Deep Residual Learning for Image Recognition, 2016.
- Cubuk et al., RandAugment: Practical Automated Data Augmentation with a Reduced Search Space, 2020.


