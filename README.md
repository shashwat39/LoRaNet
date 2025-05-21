# LoRa Signal Classification

This project implements a deep learning model for classifying LoRa signals from non-LoRa signals using spectrograms. The model is trained to distinguish between LoRa and non-LoRa signals across different Signal-to-Noise Ratio (SNR) levels.

## Project Structure

```
.
├── data/                  # Dataset directory
│   ├── train/            # Training data
│   ├── val/              # Validation data
│   └── test/             # Test data with different SNR levels
├── results/              # Training results and model outputs
│   ├── model_metrics.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── accuracy_vs_SNR.png
│   ├── learning_curves.png
│   ├── learning_rate.png
│   └── lora_classifier.onnx
├── data_gen.py           # Script for generating synthetic dataset
├── LoRaNet.py            # Main model training and evaluation script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

First, generate the synthetic dataset using `data_gen.py`:

```bash
python data_gen.py
```

This will create:
- Training and validation sets with random SNR levels (-30dB to 10dB)
- Test sets with specific SNR levels (-30dB, -20dB, -10dB, 0dB, 10dB)

### 2. Train and Evaluate Model

Run the training and evaluation script:

```bash
python LoRaNet.py
```

This will:
- Train the LoRa classifier model
- Evaluate performance across different SNR levels
- Generate performance metrics and visualizations
- Save the trained model in ONNX format

## Model Architecture

The model uses a CNN architecture with:
- 3 convolutional blocks with batch normalization
- Max pooling and global average pooling
- Dropout for regularization
- Final classification layer for binary classification

## Results

The training results are saved in the `results` directory:
- `model_metrics.txt`: Overall model performance metrics
- `classification_report.txt`: Detailed classification metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `accuracy_vs_SNR.png`: Model accuracy across different SNR levels
- `learning_curves.png`: Training and validation curves
- `learning_rate.png`: Learning rate schedule
- `lora_classifier.onnx`: Exported model in ONNX format

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python package dependencies
