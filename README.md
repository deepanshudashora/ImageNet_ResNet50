# ResNet50 ImageNet Training

Train ResNet50 model on ImageNet dataset from scratch using PyTorch.

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
albumentations>=0.5.2
numpy>=1.19.2
tensorboard>=2.4.0
```

## Dataset Setup

1. Download ImageNet dataset from [official site](https://image-net.org/download.php)
2. Extract and organize data as:

```
data/
  train/
    n01440764/
      n01440764_10026.JPEG
      ...
    ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
```

## Key Parameters

- `--data-dir`: ImageNet dataset root directory
- `--epochs`: Number of epochs (default: 90)
- `--batch-size`: Batch size per GPU (default: 256)
- `--workers`: Number of data loading workers (default: 16)
- `--lr`: Initial learning rate (default: 0.1)
- `--momentum`: SGD momentum (default: 0.9)
- `--wd`: Weight decay (default: 1e-4)

## Training Details

- Optimizer: SGD with momentum
- Learning rate schedule: Step decay

## Model Architecture

Standard ResNet50 with:

- 23.5M parameters
- 5 stages of residual blocks
- Batch normalization after each convolution
- Average pooling and 1000-way softmax
