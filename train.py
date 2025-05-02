import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from itertools import chain

from model import GPT, GPTConfig
from data import get_dataloaders
from tauon import Tauon

def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=384, help='embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--rank', type=int, default=4, help='rank for low-rank matrix factorization')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare', choices=['tiny_shakespeare', 'wikitext'], 
                        help='dataset to use')
    parser.add_argument('--block_size', type=int, default=128, help='context window size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='tokenizer type')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=15, help='total epochs to train for')
    parser.add_argument('--lr_tauon', type=float, default=0.02, help='learning rate for Tauon')
    parser.add_argument('--weight_decay_tauon', type=float, default=0.02, help='weight decay for Tauon')
    parser.add_argument('--momentum_tauon', type=float, default=0.9, help='momentum for Tauon')
    parser.add_argument('--lr_adamw', type=float, default=2e-4, help='learning rate for AdamW')
    parser.add_argument('--weight_decay_adamw', type=float, default=0.1, help='weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='device to use for training')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--eval_interval', type=int, default=200, help='evaluation interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='model saving interval')
    parser.add_argument('--output_dir', type=str, default='output', help='directory for outputs')
    
    args = parser.parse_args()
    return args

class TrainLogger:
    def __init__(self, output_dir):
        self.stats = {}
        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
        
    def log_stats(self, stats, step):
        # Update internal stats
        for k, v in stats.items():
            if k not in self.stats:
                self.stats[k] = []
            self.stats[k].append(v)
            
        # Log to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(k, v, step)
    
    def log_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)
    
    def close(self):
        self.writer.close()

def train(model, train_loader, val_loader, tauon_opt, adamw_opt, args, logger):
    """Main training loop"""
    device = args.device
    model.to(device)
    
    best_val_loss = float('inf')
    val_losses = []
    step = 0
    
    for epoch in range(args.max_epochs):
        # Training
        model.train()
        train_losses = []
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for i, (x, y) in train_pbar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping - fixed to use model parameters instead of optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer steps
            tauon_opt.step()
            adamw_opt.step()
            
            # Zero gradients
            tauon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)
            
            # Track statistics
            train_losses.append(loss.item())
            step += 1
            
            # Update progress bar
            train_pbar.set_description(f"Epoch {epoch+1}/{args.max_epochs} | Loss: {loss.item():.4f}")
            
            # Log metrics
            if i % args.log_interval == 0:
                recent_loss = np.mean(train_losses[-args.log_interval:]) if len(train_losses) >= args.log_interval else np.mean(train_losses)
                logger.log_stats({'train/loss': recent_loss, 'train/perplexity': math.exp(recent_loss)}, step)
                
            # Evaluation
            if step % args.eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                val_losses.append(val_loss)
                logger.log_stats({'val/loss': val_loss, 'val/perplexity': val_ppl}, step)
                
                # Generate sample text
                if step % (args.eval_interval * 5) == 0:
                    sample = generate_sample(model, device, max_new_tokens=100)
                    logger.log_text("samples", sample, step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, tauon_opt, adamw_opt, epoch, val_loss, os.path.join(args.output_dir, 'best_model.pt'))
                
                # Switch back to training mode
                model.train()
                
            # Save checkpoint
            if step % args.save_interval == 0:
                save_checkpoint(model, tauon_opt, adamw_opt, epoch, val_losses[-1] if val_losses else float('inf'), 
                                os.path.join(args.output_dir, f'checkpoint_{step}.pt'))
        
        # End of epoch
        epoch_loss = np.mean(train_losses)
        print(f"\nEpoch {epoch+1}/{args.max_epochs} complete | Train loss: {epoch_loss:.4f} | Train PPL: {math.exp(epoch_loss):.2f}")
                
    # Save final model
    save_checkpoint(model, tauon_opt, adamw_opt, args.max_epochs, val_losses[-1] if val_losses else float('inf'),
                    os.path.join(args.output_dir, 'final_model.pt'))
    
    return model

def evaluate(model, val_loader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
    
    val_loss = np.mean(losses)
    val_ppl = math.exp(val_loss)
    
    print(f"Validation loss: {val_loss:.4f} | Validation PPL: {val_ppl:.2f}")
    return val_loss, val_ppl

def save_checkpoint(model, tauon_opt, adamw_opt, epoch, val_loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'tauon_optimizer_state_dict': tauon_opt.state_dict(),
        'adamw_optimizer_state_dict': adamw_opt.state_dict(),
        'val_loss': val_loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")

@torch.no_grad()
def generate_sample(model, device, context=None, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate sample text from the model"""
    model.eval()
    
    if context is None:
        # Start with a random token as context
        context = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    elif isinstance(context, str) and hasattr(model, 'tokenizer'):
        # Tokenize string context
        context = torch.tensor(model.tokenizer.encode(context), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    output = model.generate(context, max_new_tokens=max_new_tokens, 
                           temperature=temperature, top_k=top_k)
    
    # Convert to text (assuming GPT-2 tokenizer for simplicity)
    if hasattr(model, 'tokenizer'):
        return model.tokenizer.decode(output[0].tolist())
    else:
        return f"Generated output ids: {output[0].tolist()}"

def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create logger
    logger = TrainLogger(args.output_dir)
    
    # Get data loaders
    train_loader, val_loader, vocab_size = get_dataloaders(
        dataset=args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        tokenizer=args.tokenizer
    )
    
    # Initialize model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rank=args.rank
    )
    model = GPT(config)
    
    # Create optimizers
    tauon_params = model.get_tauon_params()
    adamw_params = model.get_adamw_params()
    
    tauon_opt = Tauon(
        tauon_params,
        lr=args.lr_tauon, 
        weight_decay=args.weight_decay_tauon,
        momentum=args.momentum_tauon,
        nesterov=True
    )
    
    adamw_opt = AdamW(
        adamw_params,
        lr=args.lr_adamw,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay_adamw
    )
    
    # Print model and training info
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"  Tauon parameters: {sum(p.numel() for p in tauon_params)/1e6:.2f}M")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params)/1e6:.2f}M")
    print(f"Training on {args.device}")
    
    # Train the model
    start_time = time.time()
    train(model, train_loader, val_loader, tauon_opt, adamw_opt, args, logger)
    total_time = time.time() - start_time
    
    print(f"Training completed in {total_time/60:.2f} minutes")
    logger.close()

if __name__ == "__main__":
    main() 