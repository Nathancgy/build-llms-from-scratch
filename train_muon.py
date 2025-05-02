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

from model_muon import MuonGPT, MuonGPTConfig
from data import get_dataloaders
from muon import Muon

def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=384, help='embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare', choices=['tiny_shakespeare', 'wikitext'], 
                        help='dataset to use')
    parser.add_argument('--block_size', type=int, default=128, help='context window size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='tokenizer type')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=15, help='total epochs to train for')
    parser.add_argument('--lr_muon', type=float, default=0.02, help='learning rate for Muon')
    parser.add_argument('--weight_decay_muon', type=float, default=0.01, help='weight decay for Muon')
    parser.add_argument('--momentum_muon', type=float, default=0.95, help='momentum for Muon')
    parser.add_argument('--ns_steps', type=int, default=5, help='number of Newton-Schulz steps for Muon')
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
    parser.add_argument('--output_dir', type=str, default='output_muon', help='directory for outputs')
    
    # Muon specific parameters
    parser.add_argument('--world_size', type=int, default=1, help='world size for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='rank for distributed training')
    
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

def train(model, train_loader, val_loader, muon_opt, adamw_opt, args, logger):
    """Main training loop"""
    device = args.device
    model.to(device)
    
    best_val_loss = float('inf')
    val_losses = []
    step = 0
    
    # Initialize wall clock time tracking and efficiency metrics
    training_start_time = time.time()
    last_log_time = training_start_time
    tokens_processed = 0
    
    for epoch in range(args.max_epochs):
        # Training
        model.train()
        train_losses = []
        epoch_start_time = time.time()
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for i, (x, y) in train_pbar:
            batch_start_time = time.time()
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer steps
            muon_opt.step()
            adamw_opt.step()
            
            # Zero gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            # Track statistics
            train_losses.append(loss.item())
            step += 1
            
            # Update tokens processed count
            batch_tokens = x.numel()
            tokens_processed += batch_tokens
            
            # Calculate batch computational efficiency
            batch_time = time.time() - batch_start_time
            tokens_per_second = batch_tokens / batch_time
            
            # Update progress bar
            train_pbar.set_description(
                f"Epoch {epoch+1}/{args.max_epochs} | Loss: {loss.item():.4f} | {tokens_per_second:.2f} tokens/sec"
            )
            
            # Log metrics
            if i % args.log_interval == 0:
                # Calculate wall clock time metrics
                current_time = time.time()
                elapsed_time = current_time - training_start_time
                time_since_last_log = current_time - last_log_time
                last_log_time = current_time
                
                # Calculate efficiency metrics
                tokens_per_second_avg = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                recent_tokens = batch_tokens * args.log_interval
                recent_tokens_per_second = recent_tokens / time_since_last_log if time_since_last_log > 0 else 0
                
                # Log loss and perplexity
                recent_loss = np.mean(train_losses[-args.log_interval:]) if len(train_losses) >= args.log_interval else np.mean(train_losses)
                logger.log_stats({
                    'train/loss': recent_loss, 
                    'train/perplexity': math.exp(recent_loss),
                    'efficiency/tokens_per_second': tokens_per_second,
                    'efficiency/tokens_per_second_avg': tokens_per_second_avg,
                    'efficiency/recent_tokens_per_second': recent_tokens_per_second,
                    'time/elapsed_minutes': elapsed_time / 60,
                    'time/elapsed_hours': elapsed_time / 3600
                }, step)
                
            # Evaluation
            if step % args.eval_interval == 0:
                eval_start_time = time.time()
                val_loss, val_ppl = evaluate(model, val_loader, device)
                eval_time = time.time() - eval_start_time
                
                val_losses.append(val_loss)
                logger.log_stats({
                    'val/loss': val_loss, 
                    'val/perplexity': val_ppl,
                    'time/evaluation_seconds': eval_time
                }, step)
                
                # Generate sample text
                if step % (args.eval_interval * 5) == 0:
                    sample = generate_sample(model, device, max_new_tokens=100)
                    logger.log_text("samples", sample, step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, muon_opt, adamw_opt, epoch, val_loss, os.path.join(args.output_dir, 'best_model.pt'))
                
                # Switch back to training mode
                model.train()
                
            # Save checkpoint
            if step % args.save_interval == 0:
                save_checkpoint(model, muon_opt, adamw_opt, epoch, val_losses[-1] if val_losses else float('inf'), 
                                os.path.join(args.output_dir, f'checkpoint_{step}.pt'))
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        epoch_loss = np.mean(train_losses)
        epoch_tokens = len(train_loader) * args.batch_size * args.block_size
        epoch_tokens_per_second = epoch_tokens / epoch_time if epoch_time > 0 else 0
        
        logger.log_stats({
            'epoch/loss': epoch_loss,
            'epoch/perplexity': math.exp(epoch_loss),
            'epoch/time_minutes': epoch_time / 60,
            'epoch/tokens_per_second': epoch_tokens_per_second
        }, epoch)
        
        print(f"\nEpoch {epoch+1}/{args.max_epochs} complete | Train loss: {epoch_loss:.4f} | Train PPL: {math.exp(epoch_loss):.2f}")
        print(f"Epoch time: {epoch_time/60:.2f} minutes | Speed: {epoch_tokens_per_second:.2f} tokens/second")
                
    # Save final model
    save_checkpoint(model, muon_opt, adamw_opt, args.max_epochs, val_losses[-1] if val_losses else float('inf'),
                    os.path.join(args.output_dir, 'final_model.pt'))
    
    # Log final training stats
    total_training_time = time.time() - training_start_time
    total_tokens = args.max_epochs * len(train_loader) * args.batch_size * args.block_size
    overall_tokens_per_second = total_tokens / total_training_time if total_training_time > 0 else 0
    
    logger.log_stats({
        'training/total_time_hours': total_training_time / 3600,
        'training/overall_tokens_per_second': overall_tokens_per_second,
        'training/final_loss': train_losses[-1] if train_losses else 0,
        'training/best_val_loss': best_val_loss
    }, step)
    
    print(f"\nTraining completed in {total_training_time/3600:.2f} hours")
    print(f"Overall processing speed: {overall_tokens_per_second:.2f} tokens/second")
    
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

def save_checkpoint(model, muon_opt, adamw_opt, epoch, val_loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'muon_optimizer_state_dict': muon_opt.state_dict(),
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
    config = MuonGPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    )
    model = MuonGPT(config)
    
    # Create optimizers
    muon_params = model.get_muon_params()
    adamw_params = model.get_adamw_params()
    
    muon_opt = Muon(
        muon_params,
        lr=args.lr_muon, 
        weight_decay=args.weight_decay_muon,
        momentum=args.momentum_muon,
        nesterov=True,
        ns_steps=args.ns_steps,
        rank=args.rank,
        world_size=args.world_size
    )
    
    adamw_opt = AdamW(
        adamw_params,
        lr=args.lr_adamw,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay_adamw
    )
    
    # Print model and training info
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"  Muon parameters: {sum(p.numel() for p in muon_params)/1e6:.2f}M")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params)/1e6:.2f}M")
    print(f"Training on {args.device}")
    
    # Train the model
    start_time = time.time()
    train(model, train_loader, val_loader, muon_opt, adamw_opt, args, logger)
    total_time = time.time() - start_time
    
    print(f"Training completed in {total_time/60:.2f} minutes")
    logger.close()

if __name__ == "__main__":
    main() 