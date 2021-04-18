# Filter Pruning Via Softmax Attention

### Introduction

:octocat: An Official Code for **Filter Pruning Via Softmax Attention**
% # [paper](https://ieeexplore.ieee.org/document/9190715) in ICIP 2021

The goal of this tasks is efficiently pruning the network. 

### Requirements

```
- Python >= 3.5 
- Pytorch >= 1.5.0 
- torchvision >= 0.4.0 
- visdom
- numpy 
- cv2
- matplotlib
- numba
```

### Network

### Results

classification

|methods     | Dataset           |  Base-Model  | Flops  | Params   | Top-1   |Pruning_ratio|  
|------------|-------------------| ------------ | -----  | -------- |---------|-------------|
|papers      | MNIST             |  VGG5        | 12.14M |  207.19K |99.70    |0.5          |
|papers      | MNIST             |  VGG5        | 12.14M |  207.19K |99.70    |0.75         |
|papers      | MNIST             |  VGG5        | 12.14M |  207.19K |99.70    |0.875        |
|papers      | Fashion-MNIST     |  VGG5        | 32.86M |  452.04K |93.94    |0.5          |
|papers      | Fashion-MNIST     |  VGG5        | 24.82M |  351.94K |93.64    |0.75         |
|papers      | Fashion-MNIST     |  VGG5        | 12.14M |  207.19K |93.05    |0.875        |
|papers      | CIFAR10           |  VGG16       | 30.69M |  1.56M   |91.67    |0.5          |
|papers      | CIFAR10           |  VGG16       | 23.06M |  1.19M   |90.80    |0.75         |
|papers      | CIFAR10           |  VGG16       | 11.06M |  0.59M   |88.69    |0.875        |

### Quick start

#### Training

#### Testing

## Citation

