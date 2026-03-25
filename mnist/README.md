# MNIST Example

Trains a ConvNet on the MNIST dataset using PyTorch.

## Usage

### Standard (CUDA / MPS / XPU)

```bash
pip install -r requirements.txt
python main.py
# or to run on CPU only:
python main.py --no-accel
```

### TPU (via PyTorch/XLA)

To train on a Google Cloud TPU VM:

```bash
# Install PyTorch/XLA (adjust version as needed)
pip install torch torchvision
pip install 'torch_xla[tpu]'

# Run with the --xla flag
python main.py --xla
```

**Notes on TPU training:**
- The `--xla` flag selects the XLA device (TPU). It takes precedence over `torch.accelerator`.
- `drop_last=True` is used for the training DataLoader to ensure fixed batch shapes, which avoids recompilation on TPU.
- Model checkpoints saved with `--save-model` are moved to CPU before saving for cross-device portability.
- For multi-device TPU training (e.g. all 4 chips on a v4-8), see the [PyTorch/XLA multiprocessing guide](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html).

### Options

```
usage: main.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
               [--lr LR] [--gamma M] [--no-accel] [--xla] [--dry-run]
               [--seed S] [--log-interval N] [--save-model]

PyTorch MNIST Example

options:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 14)
  --lr LR              learning rate (default: 1.0)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-accel           disables accelerator
  --xla                enables XLA device (e.g. TPU). Requires torch_xla.
  --dry-run            quickly check a single pass
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         For Saving the current Model
```
