# Transformer from Scratch

This project implements a lightweight GPT-style transformer model from scratch in PyTorch for next token prediction (language modeling). The model architecture follows the decoder part of the original transformer design from the "Attention Is All You Need" paper, with adjustments to make it lightweight and runnable on an Apple M2 chip.

## Project Structure

- `model.py`: Contains the complete transformer model implementation, broken down into modular components.
- `train_lm.py`: Script for training the language model on a simple text dataset.
- `inference_lm.py`: Script for performing inference with the trained model (text generation).
- `models/`: Directory where trained models are saved.
- `data/`: Directory for text data used in training.

## Model Architecture

The model follows a GPT-style decoder-only transformer architecture with:

1. **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces.
2. **Position-wise Feed-Forward Networks**: Applied to each position separately and identically.
3. **Positional Encoding**: Adds position information to the input embeddings.
4. **Layer Normalization**: Normalizes inputs across features dimension.
5. **Residual Connections**: Helps with gradient flow and training stability.
6. **Causal Masking**: Ensures the model can only attend to previous tokens.

The implementation is broken down into the following components:

- `MultiHeadAttention`: Implements the multi-head attention mechanism.
- `PositionwiseFeedforward`: Implements the position-wise feed-forward network.
- `LayerNorm`: Implements layer normalization.
- `DecoderLayer`: Combines attention and feed-forward network for autoregressive prediction.
- `PositionalEncoding`: Adds positional information to inputs.
- `GPT`: Main model that stacks multiple decoder layers for next token prediction.

## Model Parameters

All model parameters are organized in a dedicated section at the top of `model.py` for easy access and modification:

```python
# Default values for the GPT model
DEFAULT_VOCAB_SIZE = 5000    # Vocabulary size
DEFAULT_D_MODEL = 256        # Embedding dimension
DEFAULT_NUM_HEADS = 4        # Number of attention heads
DEFAULT_D_FF = 512           # Feed-forward network dimension
DEFAULT_NUM_LAYERS = 4       # Number of transformer layers
DEFAULT_DROPOUT = 0.1        # Dropout rate
DEFAULT_MAX_LEN = 1024       # Maximum sequence length
DEFAULT_PAD_IDX = 0          # Padding token index

# Special tokens by default
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']
```

These parameters are used consistently across all scripts in the project, allowing for easy centralized configuration of the model.

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
tqdm>=4.64.0
```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Training the Model

The model is trained on a simple collection of sentences. By default, the training script will create a sample dataset of common English phrases if no other data is provided.

To train the model, run:

```
python train_lm.py
```

This will:
1. Create a simple text dataset (if not already present)
2. Initialize the transformer model
3. Train the model for the specified number of epochs
4. Save the best model based on validation loss
5. Generate sample text throughout training to show progress

The training script uses efficient dimensions to make it runnable on an Apple M2 chip:
- Model dimension (`d_model`): 256
- Number of heads (`num_heads`): 4
- Number of layers (`num_layers`): 4
- Feed-forward dimension (`d_ff`): 512

## Running Inference

To interact with the trained model and generate text, run:

```
python inference_lm.py --model_path models/gpt_best.pt
```

This starts an interactive session where you can:
- Enter prompts to generate text continuations
- Adjust generation parameters (temperature, top-k sampling)
- See the model's predictions for the next token

The inference script provides several commands:
- `/temp X`: Set the temperature to X (higher = more random)
- `/topk X`: Set the top-k sampling parameter to X (higher = more diverse)
- `/len X`: Set the maximum generation length to X tokens
- `/predict`: Show the top 5 predicted next tokens given a prompt

## Model Customization

You can customize the model by:

1. Modifying the parameters in the `# Model Configuration Parameters` section at the top of `model.py`
2. Importing and overriding specific parameters in `train_lm.py` for training experiments
3. Using your own text dataset by placing text files in the `data/` directory

To use your own dataset, place text files in the `data/` directory and modify the appropriate section in the `main()` function of `train_lm.py`.

## Learning Resources

This implementation is intended for educational purposes, allowing you to:
1. Understand the Transformer architecture by seeing each component implemented from scratch
2. Learn about language modeling and next token prediction
3. Experiment with different hyperparameters and generation strategies
4. Gain hands-on experience with PyTorch and deep learning for NLP

For more in-depth understanding, refer to:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) - A visual explanation of GPT-style models 