# NanoGPT with Neural Network Optimizers

This project implements NanoGPT-scale language models using specialized optimizers for neural networks:

1. **Tauon Optimizer**: Designed for low-rank matrix factorization, providing memory and computation efficiency while maintaining model quality.
2. **Muon Optimizer**: Momentum optimization with orthogonalization using Newton-Schulz, effective for standard neural network layers.

## Features

- NanoGPT-style transformer architecture
- Two specialized optimizer implementations:
  - **Tauon**: For low-rank factorized linear layers
  - **Muon**: For orthogonalizing standard linear layers
- Dual optimizer approach (Tauon/Muon for specialized parameters, AdamW for other parameters)
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
- `data.py`: Handles data loading and processing
- `tauon.py`: Implementation of the Tauon optimizer
- `muon.py`: Implementation of the Muon optimizer
- `train.py`: Main training script for the Tauon model
- `train_muon.py`: Training script for the Muon model
- `train_unified.py`: Unified training script that lets you choose the model type
- `generate.py`: Text generation script
- `requirements.txt`: Required packages

## Training a Model

### Training with Tauon (Low-Rank Factorization)

To train a model with Tauon optimization:

```bash
python train.py --dataset tiny_shakespeare --n_layer 6 --n_head 8 --rank 4
# or use the unified script
python train_unified.py --model_type tauon
```

### Training with Muon (Orthogonalized Momentum)

To train a model with Muon optimization:

```bash
python train_muon.py --dataset tiny_shakespeare --n_layer 6 --n_head 8 --ns_steps 5
# or use the unified script
python train_unified.py --model_type muon
```

### Key Parameters for Tauon

- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--n_embd`: Embedding dimension
- `--rank`: Rank for low-rank matrix factorization
- `--lr_tauon`: Learning rate for Tauon (default: 0.02)
- `--weight_decay_tauon`: Weight decay for Tauon (default: 0.02)
- `--momentum_tauon`: Momentum for Tauon (default: 0.9)

### Key Parameters for Muon

- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--n_embd`: Embedding dimension
- `--ns_steps`: Number of Newton-Schulz steps (default: 5)
- `--lr_muon`: Learning rate for Muon (default: 0.02)
- `--weight_decay_muon`: Weight decay for Muon (default: 0.01)
- `--momentum_muon`: Momentum for Muon (default: 0.95)
- `--world_size`: World size for distributed training (default: 1)
- `--rank_muon`: Rank for distributed training (default: 0)

### Parameter Intuition

#### Tauon
| Parameter         | Good Starting Value | Notes                            |
|-------------------|---------------------|----------------------------------|
| rank              | 4                   | Double if quality stalls early   |
| lr_tauon          | 0.02                | Scale with batch size            |
| weight_decay_tauon| 0.02                | ↑ for more aggressive rank shrink|
| momentum_tauon    | 0.9 - 0.95          | 0 → pure SGD                     |

#### Muon
| Parameter         | Good Starting Value | Notes                            |
|-------------------|---------------------|----------------------------------|
| ns_steps          | 5                   | Number of Newton-Schulz iterations|
| lr_muon           | 0.02                | Scale with batch size            |
| weight_decay_muon | 0.01                | Controls regularization          |
| momentum_muon     | 0.95                | Recommended to use high momentum |

## Generating Text

Generate text from a trained model:

```bash
python generate.py --model_path output_tauon/best_model.pt --prompt "Once upon a time" --max_new_tokens 500
# or
python generate.py --model_path output_muon/best_model.pt --prompt "Once upon a time" --max_new_tokens 500
```

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir output_tauon/logs  # For Tauon model
tensorboard --logdir output_muon/logs   # For Muon model
# Compare both models
tensorboard --logdir_spec=tauon:output_tauon/logs,muon:output_muon/logs
```

## Understanding the Optimizers

### Tauon

Tauon is a specialized optimizer for low-rank matrix factorization, based on nuclear norm regularization. It works with weights stored as `W = L @ R.T` where:

- L ∈ ℝ^(d_out×r)
- R ∈ ℝ^(d_in×r)
- r << min(d_in, d_out)

This factorization significantly reduces parameter count for large matrices.

### Muon

Muon (MomentUm Orthogonalized by Newton-schulz) uses standard SGD-momentum internally, then performs orthogonalization using Newton-Schulz iterations. This helps weight matrices maintain good conditioning throughout training, leading to more stable and often better performance.

## Citation

When using this code, please cite the original Tauon optimizer work:

```
@article{nanogpt-tauon,
  title={Efficient Training of Language Models with Low-Rank Factorization and Tauon Optimization},
  author={Your Name},
  year={2023}
}
```

## License

MIT 