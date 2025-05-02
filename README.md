# NanoGPT with Neural Network Optimizers

This project implements NanoGPT-scale language models using four different optimization approaches:

1. **Tauon Optimizer**: Uses low-rank matrix factorization, providing memory and computation efficiency while maintaining model quality.
2. **Muon Optimizer**: Applies momentum optimization with orthogonalization using Newton-Schulz, effective for standard neural network layers.
3. **Neutrino Optimizer**: Performs nuclear-norm shrinkage via randomized SVD, combining the benefits of low-rank approximation with full expressivity.
4. **Scale (Standard)**: Standard NanoGPT implementation with conventional AdamW optimizer, serving as a baseline.

## Features

- NanoGPT-style transformer architecture
- Four model optimization strategies:
  - **Tauon**: Low-rank factorized linear layers (W = L @ R.T)
  - **Muon**: Standard linear layers with orthogonalization
  - **Neutrino**: Standard linear layers with nuclear-norm proximal updates
  - **Scale**: Standard linear layers with conventional optimization
- Support for training on text datasets (tiny-shakespeare, wikitext)
- Tensorboard logging and checkpoint management
- Text generation from trained models
- Unified training script to choose between model types

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nanogpt-optimizers.git
cd nanogpt-optimizers
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Defines the GPT architecture with low-rank linear layers (for Tauon)
- `model_muon.py`: Defines the GPT architecture with standard linear layers (for Muon)
- `model_neutrino.py`: Defines the GPT architecture with standard linear layers (for Neutrino)
- `model_scale.py`: Defines the standard GPT architecture with regular linear layers
- `data.py`: Handles data loading and processing
- `tauon.py`: Implementation of the Tauon optimizer
- `muon.py`: Implementation of the Muon optimizer
- `neutrino.py`: Implementation of the Neutrino optimizer
- `train_unified.py`: Unified training script that lets you choose the model type
- `generate.py`: Text generation script
- `requirements.txt`: Required packages

## Training a Model

The unified training script lets you choose which model architecture and optimizer to use:

```bash
# Train with low-rank Tauon optimization
python train_unified.py --model_type tauon --rank 4

# Train with Muon's orthogonalized optimization
python train_unified.py --model_type muon --ns_steps 5

# Train with Neutrino's nuclear-norm proximal updates
python train_unified.py --model_type neutrino --svd_rank 6 --shrink 0.01

# Train with standard GPT (baseline)
python train_unified.py --model_type scale
```

### Common Parameters

- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--n_embd`: Embedding dimension
- `--dataset`: Dataset to use (tiny_shakespeare or wikitext)
- `--max_epochs`: Number of training epochs
- `--batch_size`: Batch size for training

### Model-Specific Parameters

#### Tauon
- `--rank`: Rank for low-rank matrix factorization
- `--lr_tauon`: Learning rate for Tauon (default: 0.02)
- `--weight_decay_tauon`: Weight decay for Tauon (default: 0.02)

#### Muon
- `--ns_steps`: Number of Newton-Schulz steps (default: 5)
- `--lr_muon`: Learning rate for Muon (default: 0.02)
- `--weight_decay_muon`: Weight decay for Muon (default: 0.01)

#### Neutrino
- `--svd_rank`: Rank for randomized SVD (default: 6)
- `--shrink`: Soft threshold for singular values (default: 0.01)
- `--power_iter`: Power iterations for randomized SVD (default: 2)
- `--lr_neutrino`: Learning rate for Neutrino (default: 0.02)
- `--weight_decay_neutrino`: Weight decay for Neutrino (default: 0.01)

#### Scale (Standard)
- `--lr_adamw`: Learning rate for AdamW (default: 2e-4)
- `--weight_decay_adamw`: Weight decay for AdamW (default: 0.1)

## Comparing Performance

You can use TensorBoard to compare the performance of the four approaches:

```bash
# Compare all model types
tensorboard --logdir_spec=tauon:output_tauon/logs,muon:output_muon/logs,neutrino:output_neutrino/logs,scale:output_scale/logs
```

This will allow you to visualize:
- Training and validation loss/perplexity
- Training speed (tokens per second)
- Memory efficiency
- Convergence rates

## Understanding the Optimizers

### Tauon

Tauon is a specialized optimizer for low-rank matrix factorization, based on nuclear norm regularization. It works with weights stored as `W = L @ R.T` where:

- L ∈ ℝ^(d_out×r)
- R ∈ ℝ^(d_in×r)
- r << min(d_in, d_out)

This factorization significantly reduces parameter count for large matrices, trading off some model capacity for efficiency.

### Muon

Muon (MomentUm Orthogonalized by Newton-schulz) uses standard SGD-momentum internally, then performs orthogonalization using Newton-Schulz iterations. This helps weight matrices maintain good conditioning throughout training, leading to more stable optimization and often better performance with full parameter capacity.

### Neutrino

Neutrino (Nuclear-Norm Proximal SGD) works with full matrices like Muon, but instead of orthogonalizing, it applies nuclear-norm regularization through a randomized SVD. It:

1. Computes momentum updates like Muon
2. Performs a randomized truncated SVD to get U, S, V^T
3. Soft-thresholds the singular values: S' = max(S - λ, 0)
4. Reconstructs the low-rank update as U S' V^T

This approach allows for pruning small singular values while maintaining computational efficiency, offering a middle ground between full expressivity and low-rank approximation.

### Scale (Standard)

The standard model uses conventional AdamW optimization without parameter factorization or orthogonalization. It maintains full expressivity but doesn't benefit from the specialized optimization strategies of the other approaches.

## License

MIT 