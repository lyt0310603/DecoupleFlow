# DecoupleFlow User Guide

## DecoupleFlow Overview

### What is DecoupleFlow?

DecoupleFlow is a Python package that converts your existing model (usually a layer sequence built with `torch.nn.Sequential`) into a block-wise training architecture. Each block can be updated with its own local objective, and blocks can be placed on different devices (CPU / multiple GPUs).

### Installation Requirements and Setup

- **Python**: 3.9 or above (including 3.9)
- **PyTorch**: `torch>=2.0` (CUDA support depends on your installed torch build)

The project is hosted on GitHub, and the package name in `pyproject.toml` is `decoupleflow`. For development and experiments, the recommended workflow is: clone from GitHub + editable install (`pip install -e .`).

1. **Download from GitHub**

   If `git` is installed, cloning is recommended:

   ```bash
   # Download source code
   git clone https://github.com/lyt0310603/DecoupleFlow.git

   # Enter the project root (the directory containing pyproject.toml)
   cd DecoupleFlow
   ```

   If you do not have `git`, or you are in a restricted environment:

   - Go to the GitHub project page (`https://github.com/lyt0310603/DecoupleFlow`)
   - Click "Code" -> "Download ZIP"
   - Extract the ZIP and open a terminal in the extracted folder

2. **Install with `pip install -e .`**

   Run this in the project root (same level as `pyproject.toml`):

   ```bash
   pip install -e .
   ```

   - This installs `decoupleflow` in editable mode, so source code changes in this folder are reflected immediately.
   - After installation, you can import it in your environment:

   ```python
   from decoupleflow import DecoupleFlow
   ```

## Core Concepts

- **Block Partitioning**: DecoupleFlow splits your `custom_model` into multiple blocks based on `device_map`. Each block is wrapped as a `BasicBlock` (or `AdaptiveBasicBlock` in adaptive mode).
- **Device Assignment**: Each block can run on a different device (for example `cuda:0`, `cuda:1`). Gradients are isolated between blocks using `detach()`, which is central to decoupled training.
- **Local Loss**: Intermediate blocks can use `CL` (Supervised Contrastive) or `DeInfo` (information regularization). In non-adaptive mode, the final block is forced to use `CE` (CrossEntropy).
- **PyTorch-like Training Loop**: You can still use `model.train()` / `model.eval()` and call `model(X, Y)` each step for training/inference.

![DecoupleFlow partition](fig/partirion.png)
*The figure above illustrates DecoupleFlow partitioning strategy: the left side is a model built in PyTorch, and the right side is the model after DecoupleFlow refactoring.*

## Input / Output

### Training

During training, the recommended call is `model(X, Y)` (under `model.train()`, it is automatically dispatched to the training path):

- **Input**: `X`, `Y`, optional `mask`
- **Output**: `features, total_loss, labels`
  - `features`: output feature from the last block (usually logits or features depending on your backbone's final layer)
  - `total_loss`: sum of local losses from all blocks, returned as Python `float`
  - `labels`: labels moved to the last block's device

```python
model.train()
features, total_loss, labels = model(X, Y, mask=mask)
```

If you want to explicitly separate training and inference functions, you can call `train_step` directly:

```python
model.train()
features, total_loss, labels = model.train_step(X, Y, mask=mask)
```

### Testing / Inference

During testing, the recommended call is also `model(X, Y)` (under `model.eval()`, it is automatically dispatched to the inference path):

- **Input**: `X`, `Y`, optional `mask`
- **Output**: `output, labels`

```python
model.eval()
output, labels = model(X, Y, mask=mask)
```

If you want to explicitly choose the inference function, you can call `test_step` directly:

```python
model.eval()
output, labels = model.test_step(X, Y, mask=mask)
```

### Adaptive Inference Mode (Early Exit)

When `is_adaptive=True`, calling `model(X, Y)` under `model.eval()` is dispatched to adaptive inference and returns:

- **Output**: `classifier_output, stop_layer_index, labels`

```python
model.eval()
classifier_output, stop_layer, labels = model(X, Y, mask=mask)
```

You can also call `adaptive_test_step` explicitly:

```python
model.eval()
classifier_output, stop_layer, labels = model.adaptive_test_step(X, Y, mask=mask)
```

> Note: In adaptive mode, each block has an extra classifier (see `classifier` parameter). Early exit is decided by cosine similarity and argmax consistency between adjacent block logits.

## Parameter Reference (for `DecoupleFlow`)

Required parameters are `custom_model` and `device_map`. Common optional parameters include `loss_fn`, `num_classes`, `projector_type`, `custom_projector`, `transform_funcs`, `optimizer_fn`, `optimizer_param`, `scheduler_fn`, `scheduler_param`, `multi_t`, `is_adaptive`, `patiencethreshold`, `cosinesimthreshold`, and `classifier`.

### custom_model

**Purpose**: Provides the backbone model that DecoupleFlow will split into multiple blocks.
- **Recommendation**: use `torch.nn.Sequential` to define a model that can be partitioned.

```python
import torch.nn as nn

backbone = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
)
```

You can also place custom class-based modules inside `Sequential`.  
Pay attention to the custom module `forward` return format, since it affects local loss computation:

- If no extra handling is needed, return a single output `result`.
- If you need a dedicated feature for local loss, return `(result, for_loss)`.
  - For example, in sequence models you may use a pooled feature (such as averaged hidden states) for loss.
  - If this custom layer is the last layer of a block, `for_loss` will be used for local loss.

> Note: If you directly use `nn.LSTM`, DecoupleFlow already has built-in handling to extract local-loss features. A second return value is only needed for custom layers with special requirements.

Simple example:

```python
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, in_dim=16, out_dim=8):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        result = self.fc(x)
        for_loss = result.mean(dim=1)  # example: dedicated feature for local loss
        return result, for_loss

model = nn.Sequential(
    nn.Linear(32, 16),
    CustomLayer(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
```

### device_map

**Purpose**: Defines how many layers each block contains, and which device each block is assigned to.
You can use two formats:

1) **dict format**: `{device: layers_count, ...}`

```python
device_map = {"cuda:0": 2, "cuda:1": 2, "cuda:2": 1}
```

2) **list format**: each segment is `{"device": "...", "layers": ...}` (useful when multiple segments are placed on the same device)

```python
device_map = [
    {"device": "cpu", "layers": 1},
    {"device": "cpu", "layers": 1},
    {"device": "cpu", "layers": 1},
]
```

`layers` means **cumulative sequential partitioning**, not absolute layer indices.  
DecoupleFlow starts from layer 1 of `custom_model` and assigns layers continuously in the order of `device_map`:

- Segment 1 takes the first `layers` layers
- Segment 2 takes the next `layers` layers
- And so on until all layers are assigned

For example, with `device_map = {"cuda:0": 2, "cuda:1": 2, "cuda:2": 1}`:

- `cuda:0` gets layers 1-2
- `cuda:1` gets layers 3-4
- `cuda:2` gets layer 5

Minimal readable example (3-layer model):

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

device_map = {"cuda:0": 2, "cuda:1": 1}
```

Advanced example for NLP-style structure:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Embedding(num_embeddings=5000, embedding_dim=128),
    nn.LSTM(input_size=128, hidden_size=256, batch_first=True),
    nn.LSTM(input_size=256, hidden_size=256, batch_first=True),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128, 2),
)

device_map = {
    "cuda:0": 1,  # assign Embedding to GPU 0
    "cuda:1": 1,  # assign first LSTM to GPU 1
    "cuda:2": 1,  # assign second LSTM to GPU 2
    "cuda:3": 3,  # assign remaining layers to GPU 3
}
```

Important constraints:

- `device_map` **cannot be None**
- `sum(layers)` must equal `len(custom_model)` (number of layers in `Sequential`)
- each item's `layers` must be `> 0`
- each item's `device` cannot be `None`

### loss_fn

**Purpose**: Specifies the local loss type for blocks, controlling how each block learns.
`loss_fn` currently **must be a string**:

- `"CL"`: Supervised Contrastive Loss (default for intermediate blocks)
- `"DeInfo"`: DeInfo Loss (requires `num_classes`)

> In non-adaptive mode, the final block is automatically set to `"CE"`, while intermediate blocks use your `loss_fn`.

### num_classes

**Purpose**: Provides class count so `DeInfo` loss and adaptive classifier can use the correct output dimension.
`num_classes` means the number of label classes in your dataset (for example, binary classification is `2`, MNIST is `10`).
`num_classes` is required when:

- you use `loss_fn="DeInfo"`
- adaptive mode with `classifier=None` (using built-in classifier)

> If you pass your own `classifier`, your classifier architecture is used as-is. The package will not rebuild your classifier based on `num_classes`.

### projector_type and custom_projector

**Purpose (`projector_type`)**: Chooses which projection head is used before local loss in each block.  
**Purpose (`custom_projector`)**: Lets you provide your own projection module when `projector_type="c"`.
`projector_type` supports:

- `"i"`: Identity (no dimension change)
- `"l"`: single Linear (lazy)
- `"mlp"`: two-layer MLP (lazy -> 512 -> 1024)
- `"DeInfo"`: DeInfo projector (deeper MLP)
- `"c"`: custom projector (pass a `torch.nn.Module` via `custom_projector`)

> Note: Built-in projectors use `nn.LazyLinear` heavily because after block partitioning, each block may output different dimensions, and `in_features` may be unknown at initialization.  
> Lazy layers infer input dimensions on first forward, reducing manual shape calculation and maintenance effort.

Custom projector example:

```python
import torch.nn as nn

custom_projector = nn.Sequential(
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.Linear(128, 64),
)

model = DecoupleFlow(
    custom_model=backbone,
    device_map=device_map,
    loss_fn="CL",
    num_classes=4,
    projector_type="c",
    custom_projector=custom_projector,
)
```

### transform_funcs (inter-block transform functions)

**Purpose**: Controls feature transformation between blocks so the next block receives input in the correct format.
`transform_funcs` is a list whose length must equal the number of blocks (segments split by `device_map`). Each element can be:

- `callable`: converts previous block output (list of tensors) into input format required by the next block encoder
- `None`: uses default transform (take the first input tensor)

Why is this needed?

- Inside `BasicBlock`, outputs are passed as a list. If your layer returns complex outputs (for example RNN/Transformer style), you may need to manually select the tensor fed into the next block.

In general, when connecting LSTM to a linear layer, you take `x[:, -1, :]`, i.e., the last time-step output. First, here is the standard PyTorch style:

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm1 = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x[:, -1, :])  # use the last output of LSTM
        x = self.tanh(x)
        x = self.fc2(x)
        return x
```

In DecoupleFlow, because the model is split into multiple blocks, if the split point is between LSTM and Linear, you still need to keep this "take last output" operation. So use the second style: explicitly add a transform function through `transform_funcs` to avoid shape mismatch between LSTM and Linear.

The following example is based on the earlier `backbone` and `device_map` settings (where LSTM and classifier layers are in different blocks):

```python
# LSTM -> classifier requires transformation
# nn.LSTM returns x, (h, c), so function parameters are defined as x, h, c
def LSTMtoLinear(x, h, c):
    return x[:, -1, :]

# use None where no transform function is needed
# previous example splits model into 4 blocks, so transform_funcs length must be 4
# insert transform between block 3 (LSTM) and block 4 (Linear)
transform_funcs = [None, None, None, LSTMtoLinear]
```

![transform_funcs](fig/transform.png)
*The figure above shows how transform functions are inserted between blocks in DecoupleFlow.*

### optimizer_fn / optimizer_param

**Purpose (`optimizer_fn`)**: Specifies optimizer class used by each block.  
**Purpose (`optimizer_param`)**: Provides optimizer hyperparameters (for example `lr`, `momentum`).
  - Passed optimizer hyperparameters must match the selected optimizer's requirements.
DecoupleFlow creates one optimizer per block. You only need to pass optimizer class and parameter dict:

```python
import torch

optimizer_fn = torch.optim.SGD
optimizer_param = {"lr": 0.01, "momentum": 0.9}
```

### scheduler_fn / scheduler_param

**Purpose (`scheduler_fn`)**: Specifies learning rate scheduler type for each block.  
**Purpose (`scheduler_param`)**: Provides scheduler initialization parameters.
  - Passed scheduler hyperparameters must match the selected scheduler's requirements.
If you provide `scheduler_fn`, you must also provide non-empty `scheduler_param`. Each block gets its own scheduler.

Configuration example:

```python
import torch

scheduler_fn = torch.optim.lr_scheduler.StepLR
scheduler_param = {
    "step_size": 10,
    "gamma": 0.2,
}
```

When you want to update schedulers, call `scheduler_step()`. DecoupleFlow will update all block schedulers together:

```python
model.scheduler_step()
```

### multi_t

**Purpose**: Controls whether each block's `loss + backward + optimizer.step` runs in parallel using multithreading.
`multi_t` is a boolean and is enabled by default:

- `True`: enable multithreaded parallel execution
- `False`: run sequentially (often used in tests for controlled behavior)

### is_adaptive / patiencethreshold / cosinesimthreshold / classifier

**Purpose (`is_adaptive`)**: Enables adaptive inference (early exit) flow.  
**Purpose (`patiencethreshold`)**: Sets how many consecutive early-exit conditions are required.  
**Purpose (`cosinesimthreshold`)**: Sets similarity threshold between adjacent layer prediction vectors.  
**Purpose (`classifier`)**: Specifies auxiliary classifier used by each block in adaptive mode.

- `is_adaptive=True` enables adaptive inference
- `patiencethreshold`: stop after condition is met continuously (at least 1)
- `cosinesimthreshold`: cosine similarity threshold between adjacent layer logits (float)
- `classifier`: optional custom classifier (`nn.Module`); if not provided, built-in default classifier is used (see `ExtraLayer`)

## Common Error Troubleshooting

- **`Layers of model don't equal to balance`**: total layer counts in `device_map` do not equal `len(custom_model)`.
- **`device_map cannot be None`**: you did not provide `device_map`.
- **`Cannot distribute transform_funcs`**: length of `transform_funcs` does not match number of blocks.
- **`DeInfo Loss need pass class nums`**: you must provide `num_classes` when `loss_fn="DeInfo"`.
