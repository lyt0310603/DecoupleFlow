# DecoupleFlow

DecoupleFlow is a PyTorch-based toolkit for converting a user-defined model into a decoupled training architecture.  
It focuses on practical implementations of SCPL-style and DeInfoReg-style workflows, with model partitioning and multi-device execution support for efficient training.

## Project Status

This project is under active development.  
APIs, behavior, and documentation may change between versions.

## What This Package Solves

Decoupled learning methods can improve training efficiency, but they usually require significant refactoring of model code.  
DecoupleFlow reduces this engineering overhead by providing a configurable framework that:

- partitions models into training blocks,
- assigns blocks across devices,
- attaches local objectives to intermediate blocks, and
- keeps the training loop close to common PyTorch workflows.

## Key Features

- **Model Refactoring for Decoupled Training**: Transform a standard model into block-wise training components.
- **Pipeline-Style Partitioning**: Split models into sequential blocks and distribute them across GPUs.
- **Local Objective Support**: Enable local losses (for example, contrastive-style objectives) at block level.
- **Flexible Configuration**: Control partitioning, optimization setup, and execution behavior through parameters.
- **Adaptive Extensions**: Support adaptive inference related features for decoupled architectures.

## Method Background

### SCPL
Supervised Contrastive Parallel Learning (SCPL) decouples end-to-end backpropagation using local objectives and supervised contrastive learning, reducing backward dependency across the whole network.  
Reference implementation: [SCPL GitHub](https://github.com/minyaho/SCPL/tree/main?tab=readme-ov-file)

### DeInfoReg
Decoupled Supervised Learning with Information Regularization (DeInfoReg) extends decoupled learning with information-regularized local objectives and adaptive architectural ideas.  
Reference implementation: [DeInfoReg GitHub](https://github.com/ianzih/Decoupled-Supervised-Learning-for-Information-Regularization/tree/master)

## Quick Start

### 1) Download and install DecoupleFlow

```bash
git clone https://github.com/lyt0310603/DecoupleFlow.git
pip install -e .
```

### 2) Minimal usage example

```python
import torch
import torch.nn as nn
from decoupleflow import DecoupleFlow

# Define a simple backbone model.
backbone = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
)

# Partition layers into blocks and map blocks to devices.
device_map = {"cuda:0": 2, "cuda:1": 2, "cuda:2": 1}

model = DecoupleFlow(
    custom_model=backbone,
    device_map=device_map,
    loss_fn="CL",              # or "DeInfo"
    projector_type="i",        # Identity projector
    optimizer_fn=torch.optim.Adam,
    optimizer_param={"lr": 1e-3},
    multi_t=True,
    is_adaptive=False,
)

x = torch.randn(32, 768)
y = torch.randint(0, 4, (32,))

features, loss, labels = model.train_step(x, y)
output, labels = model.test_step(x, y)
```

## Documentation
- Detailed usage instructions (Chinese): [Here](https://github.com/lyt0310603/DecoupleFlow/blob/main/docs/tutorial_ch.md)
- Detailed usage instructions (English): [Here](https://github.com/lyt0310603/DecoupleFlow/blob/main/docs/tutorial_en.md)


