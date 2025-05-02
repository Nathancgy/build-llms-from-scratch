# NanoGPT with Tauon Optimizer

This project implements a NanoGPT-scale language model using the Tauon optimizer for low-rank matrix factorization. Tauon is a specialized optimizer that works with low-rank factorized neural network weights, providing memory and computation efficiency while maintaining model quality.

## Features

- NanoGPT-style transformer architecture
- Low-rank linear layers with Tauon optimization
- Dual optimizer approach (Tauon for low-rank factors, AdamW for other parameters)
- Support for training on text datasets (tiny-shakespeare, wikitext)
- Tensorboard logging and checkpoint management
- Text generation from trained models

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nanogpt-tauon.git
cd nanogpt-tauon
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Defines the GPT architecture with low-rank linear layers
- `data.py`: Handles data loading and processing
- `tauon.py`: Implementation of the Tauon optimizer
- `train.py`: Main training script
- `generate.py`: Text generation script
- `requirements.txt`: Required packages

## Training a Model

To train a model with default settings (tiny-shakespeare dataset, 6 layers, 8 heads, rank-4 factorization):

```bash
python train.py
```

Customize training with command-line arguments:

```bash
python train.py --dataset wikitext --n_layer 8 --n_head 12 --n_embd 512 --rank 8 --batch_size 32
```

### Key Parameters

- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--n_embd`: Embedding dimension
- `--rank`: Rank for low-rank matrix factorization
- `--lr_tauon`: Learning rate for Tauon (default: 0.02)
- `--weight_decay_tauon`: Weight decay for Tauon (default: 0.02)
- `--lr_adamw`: Learning rate for AdamW (default: 2e-4)
- `--dataset`: Dataset to use (tiny_shakespeare or wikitext)

### Parameter Intuition

| Parameter         | Good Starting Value | Notes                            |
|-------------------|---------------------|----------------------------------|
| rank              | 4                   | Double if quality stalls early   |
| lr_tauon          | 0.02                | Scale with batch size            |
| weight_decay_tauon| 0.02                | ↑ for more aggressive rank shrink|
| momentum_tauon    | 0.9 - 0.95          | 0 → pure SGD                     |

## Generating Text

Generate text from a trained model:

```bash
python generate.py --model_path output/best_model.pt --prompt "Once upon a time" --max_new_tokens 500
```

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir output/logs
```

## Understanding Tauon

Tauon is a specialized optimizer for low-rank matrix factorization, based on nuclear norm regularization. It works with weights stored as `W = L @ R.T` where:

- L ∈ ℝ^(d_out×r)
- R ∈ ℝ^(d_in×r)
- r << min(d_in, d_out)

This factorization significantly reduces parameter count for large matrices.

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