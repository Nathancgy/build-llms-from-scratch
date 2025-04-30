import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
import re
from tqdm import tqdm

from model import GPT, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, SPECIAL_TOKENS
from model import DEFAULT_VOCAB_SIZE, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS
from model import DEFAULT_D_FF, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT, DEFAULT_MAX_LEN

# =============================================================================
# Training Configuration
# =============================================================================

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 15

# Use default model hyperparameters from model.py
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
D_MODEL = DEFAULT_D_MODEL
NUM_HEADS = DEFAULT_NUM_HEADS
D_FF = DEFAULT_D_FF
NUM_LAYERS = DEFAULT_NUM_LAYERS
DROPOUT = DEFAULT_DROPOUT
MAX_LEN = DEFAULT_MAX_LEN

# =============================================================================
# Dataset and Data Loading
# =============================================================================

class SimpleTextDataset(Dataset):
    """
    A simple dataset for text data with basic preprocessing.
    """
    def __init__(self, text_files, max_len, vocab=None, min_freq=2):
        self.max_len = max_len
        
        # Load and preprocess text
        self.raw_text = []
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.raw_text.extend(f.readlines())
        
        # Preprocess the text
        self.examples = []
        for line in tqdm(self.raw_text, desc="Preprocessing text"):
            # Clean and tokenize the text
            line = line.strip().lower()
            if len(line) > 0:
                # Simple tokenization (split by whitespace and punctuation)
                tokens = re.findall(r'\b\w+\b|[^\w\s]', line)
                if len(tokens) > 2:  # Ensure we have at least a few tokens
                    self.examples.append(tokens)
        
        # Build vocabulary if not provided
        if vocab is None:
            # Count word frequencies
            counter = Counter()
            for example in self.examples:
                counter.update(example)
            
            # Create vocabulary with words above min frequency
            self.vocab = SPECIAL_TOKENS.copy()
            self.vocab.extend([word for word, count in counter.most_common(VOCAB_SIZE - len(SPECIAL_TOKENS)) 
                              if count >= min_freq])
            
            # Create word to index mapping
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        else:
            self.vocab = vocab
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Convert tokens to indices
        indices = [BOS_IDX]
        for token in tokens[:self.max_len-2]:  # -2 for BOS and EOS
            indices.append(self.word2idx.get(token, UNK_IDX))
        indices.append(EOS_IDX)
        
        return torch.LongTensor(indices)


def collate_batch(batch):
    """
    Collate function for DataLoader.
    """
    # Pad the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    
    # For language modeling, the input is the sequence except the last token
    # and the target is the sequence except the first token
    src = padded_batch[:, :-1]
    tgt = padded_batch[:, 1:]
    
    return src, tgt

# =============================================================================
# Model Training and Evaluation
# =============================================================================

def initialize_model():
    """
    Initialize the model, loss function, and optimizer.
    """
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX
    ).to(device)
    
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Define loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, verbose=False
    )
    
    return model, criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_loss = 0
    total_batches = len(dataloader)
    
    # Create progress bar
    progress_bar = tqdm(total=total_batches, desc="Training")
    
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(src)
        
        # Reshape for loss calculation
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        logits = logits.contiguous().view(-1, logits.shape[-1])
        # [batch_size, seq_len] -> [batch_size * seq_len]
        tgt = tgt.contiguous().view(-1)
        
        # Compute loss
        loss = criterion(logits, tgt)
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar with current loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        progress_bar.update(1)
    
    progress_bar.close()
    return epoch_loss / total_batches


def evaluate(model, dataloader, criterion):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    epoch_loss = 0
    total_batches = len(dataloader)
    
    # Create progress bar
    progress_bar = tqdm(total=total_batches, desc="Evaluating")
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Forward pass
            logits, _ = model(src)
            
            # Reshape for loss calculation
            logits = logits.contiguous().view(-1, logits.shape[-1])
            tgt = tgt.contiguous().view(-1)
            
            # Compute loss
            loss = criterion(logits, tgt)
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.update(1)
    
    progress_bar.close()
    return epoch_loss / total_batches


def generate_text(model, vocab, prompt="", max_len=50, temperature=1.0, top_k=None):
    """
    Generate text from the model.
    """
    model.eval()
    
    # Convert words to indices
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Tokenize the prompt if provided
    if prompt:
        tokens = prompt.strip().lower().split()
        input_ids = [BOS_IDX]
        for token in tokens:
            input_ids.append(word2idx.get(token, UNK_IDX))
    else:
        input_ids = [BOS_IDX]
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            prompt=input_tensor,
            max_new_tokens=max_len,
            temperature=temperature,
            top_k=top_k
        )
    
    # Convert indices back to words
    generated_text = []
    for idx in generated[0]:
        if idx.item() == BOS_IDX:
            continue
        if idx.item() == EOS_IDX:
            break
        generated_text.append(vocab[idx.item()])
    
    return " ".join(generated_text)

# =============================================================================
# Data Generation and Training Loop
# =============================================================================

def create_sample_dataset():
    """
    Create a sample text dataset of common English phrases.
    """
    # Set up data files
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample text if it doesn't exist
    sample_file = os.path.join(data_dir, "sample_text.txt")
    if not os.path.exists(sample_file):
        # Create a simple text file with some random sentences
        print("Creating sample text file...")
        sample_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "Actions speak louder than words.",
            "Knowledge is power.",
            "Time flies like an arrow; fruit flies like a banana.",
            "You can't teach an old dog new tricks.",
            "The pen is mightier than the sword.",
            "Where there's a will, there's a way.",
            "Birds of a feather flock together.",
            "Two wrongs don't make a right.",
            "The early bird catches the worm.",
            "Better late than never.",
            "Beauty is in the eye of the beholder.",
            "You can't judge a book by its cover.",
            "Necessity is the mother of invention.",
            "Fortune favors the bold.",
            "Practice makes perfect.",
            "Don't count your chickens before they hatch.",
            "There's no place like home.",
            "When in Rome, do as the Romans do.",
            "Look before you leap.",
            "Too many cooks spoil the broth.",
            "Curiosity killed the cat.",
            "The grass is always greener on the other side.",
            "We learn from our mistakes.",
            "Rome wasn't built in a day.",
            "A stitch in time saves nine.",
            "The apple doesn't fall far from the tree.",
            "You can lead a horse to water but you can't make it drink.",
            "Great minds think alike.",
            "Don't put all your eggs in one basket.",
            "All good things must come to an end.",
        ]
        
        # Repeat and permute sentences to create a larger dataset
        expanded_sentences = []
        for i in range(30):  # Repeat with variations
            for sentence in sample_sentences:
                # Sometimes add a follow-up sentence
                if np.random.random() < 0.3:
                    idx = np.random.randint(0, len(sample_sentences))
                    expanded_sentences.append(sentence + " " + sample_sentences[idx])
                else:
                    expanded_sentences.append(sentence)
        
        # Shuffle the expanded sentences
        np.random.shuffle(expanded_sentences)
        
        # Write to file
        with open(sample_file, 'w', encoding='utf-8') as f:
            for sentence in expanded_sentences:
                f.write(sentence + "\n")
    
    # Split data into train and validation sets
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print("Splitting data into train and validation sets...")
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Shuffle the lines
        np.random.shuffle(lines)
        
        # Split into train (80%) and validation (20%)
        split_idx = int(0.8 * len(lines))
        train_lines = lines[:split_idx]
        valid_lines = lines[split_idx:]
        
        # Write to files
        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        
        with open(valid_file, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
    
    return train_file, valid_file


def main():
    """
    Main training loop.
    """
    print("Starting language model training...")
    
    # Create sample dataset and get file paths
    train_file, valid_file = create_sample_dataset()
    
    # Load dataset
    train_dataset = SimpleTextDataset([train_file], max_len=MAX_LEN)
    valid_dataset = SimpleTextDataset([valid_file], max_len=MAX_LEN, vocab=train_dataset.vocab)
    
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(valid_dataset)}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Initialize model, loss, optimizer
    model, criterion, optimizer, scheduler = initialize_model()
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    best_valid_loss = float('inf')
    print("Starting training...")
    
    # Create a wrapper progress bar for the epochs
    epoch_progress = tqdm(range(NUM_EPOCHS), desc="Epochs")
    
    for epoch in epoch_progress:
        start_time = time.time()
        
        # Train and evaluate
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        valid_loss = evaluate(model, valid_dataloader, criterion)
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Calculate time elapsed
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Update epoch progress bar description
        epoch_progress.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_progress.set_postfix({
            "Train Loss": f"{train_loss:.4f}", 
            "Valid Loss": f"{valid_loss:.4f}",
            "Time": f"{epoch_mins}m {epoch_secs:.0f}s"
        })
        
        # Save model if validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'vocab': train_dataset.vocab,
                'hyperparams': {
                    'vocab_size': VOCAB_SIZE,
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'd_ff': D_FF,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'max_len': MAX_LEN
                }
            }, 'models/gpt_best.pt')
            # Update on best model save
            tqdm.write(f"Epoch {epoch+1}: New best model saved (valid loss: {valid_loss:.4f})")
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
        'vocab': train_dataset.vocab,
        'hyperparams': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'd_ff': D_FF,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'max_len': MAX_LEN
        }
    }, 'models/gpt_final.pt')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print("Final model saved to models/gpt_final.pt")


if __name__ == "__main__":
    main() 