import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import requests
from tqdm import tqdm
import tiktoken

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def download_tiny_shakespeare():
    """Downloads the tiny shakespeare dataset if not already present"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, 'tinyshakespeare.txt')
    if not os.path.exists(filepath):
        print("Downloading tiny shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(requests.get(url).text)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

def download_wikitext():
    """Downloads the wikitext-103 dataset if not already present"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, 'wikitext-103-sample.txt')
    
    if not os.path.exists(filepath):
        print("Downloading WikiText-103 sample...")
        url = 'https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/wikitext-103/wiki.train.tokens'
        response = requests.get(url, stream=True)
        
        # Get content length if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Take just the first 5MB of data as a sample
            bytes_written = 0
            max_bytes = 5 * 1024 * 1024  # 5MB
            
            with tqdm(total=min(total_size, max_bytes), unit='iB', unit_scale=True) as t:
                for data in response.iter_content(block_size):
                    bytes_written += len(data)
                    f.write(data.decode('utf-8'))
                    t.update(len(data))
                    if bytes_written >= max_bytes:
                        break
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

def prepare_data(text, tokenizer_name="gpt2", train_split=0.9):
    """
    Prepares data for training by tokenizing and splitting into train/val
    
    Args:
        text: raw text data
        tokenizer_name: name of the tokenizer to use (default: gpt2)
        train_split: fraction of data to use for training (default: 0.9)
    
    Returns:
        train_data, val_data as tokenized numpy arrays
    """
    # Initialize tokenizer
    if tokenizer_name == "gpt2":
        print("Using GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
    else:
        # Character-level tokenization as fallback
        print("Using character-level tokenization")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print(f"Vocabulary size: {vocab_size}")
        stoi = {ch: i for i, ch in enumerate(chars)}
        tokens = [stoi[c] for c in text]
    
    tokens = np.array(tokens, dtype=np.int32)
    
    # Split into train and validation
    n = len(tokens)
    train_tokens = tokens[:int(n * train_split)]
    val_tokens = tokens[int(n * train_split):]
    
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    
    return train_tokens, val_tokens

def get_dataloaders(dataset='tiny_shakespeare', block_size=128, batch_size=64, tokenizer="gpt2"):
    """
    Creates the data loaders for training
    
    Args:
        dataset: name of the dataset to use ('tiny_shakespeare' or 'wikitext')
        block_size: size of the context window
        batch_size: batch size for training
        tokenizer: tokenizer type to use
    
    Returns:
        train_loader, val_loader, vocab_size
    """
    # Get the raw text based on dataset choice
    if dataset == 'tiny_shakespeare':
        text = download_tiny_shakespeare()
    elif dataset == 'wikitext':
        text = download_wikitext()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Prepare the data
    train_data, val_data = prepare_data(text, tokenizer_name=tokenizer)
    
    # Create the datasets
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Determine vocab size
    if tokenizer == "gpt2":
        vocab_size = 50257  # GPT-2 vocabulary size
    else:
        # For character-level, count unique tokens
        vocab_size = max(np.max(train_data), np.max(val_data)) + 1
    
    return train_loader, val_loader, vocab_size 