import torch
import argparse
import os
import re
from tqdm import tqdm

from model import GPT, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX

# =============================================================================
# Inference Utilities
# =============================================================================

class LanguageModelInference:
    """
    A class for performing inference with a trained language model.
    """
    def __init__(self, model_path, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load saved model
        print("Loading model...")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model hyperparameters
        hyperparams = self.checkpoint['hyperparams']
        
        # Initialize model
        self.model = GPT(
            vocab_size=hyperparams['vocab_size'],
            d_model=hyperparams['d_model'],
            num_heads=hyperparams['num_heads'],
            d_ff=hyperparams['d_ff'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
            max_len=hyperparams['max_len'],
            pad_idx=PAD_IDX  # Using imported PAD_IDX constant
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get vocabulary
        self.vocab = self.checkpoint['vocab']
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Special token indices are imported from model.py
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX
        
        print(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def generate_text(self, prompt="", max_len=50, temperature=0.7, top_k=40, show_progress=False):
        """
        Generate text given a prompt.
        
        Args:
            prompt (str): The prompt text to start generation from
            max_len (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling (higher = more random)
            top_k (int): If set, only sample from the top k most likely tokens
            show_progress (bool): Whether to show a progress bar during generation
            
        Returns:
            str: The generated text
        """
        self.model.eval()
        
        # Tokenize the prompt if provided
        if prompt:
            # Simple tokenization by whitespace and punctuation
            tokens = re.findall(r'\b\w+\b|[^\w\s]', prompt.strip().lower())
            input_ids = [self.BOS_IDX]
            for token in tokens:
                input_ids.append(self.word2idx.get(token, self.UNK_IDX))
        else:
            input_ids = [self.BOS_IDX]
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        
        # Generate text
        with torch.no_grad():
            # Start with the prompt
            x = input_tensor.clone()
            
            # Create progress bar if requested
            progress_iter = tqdm(range(max_len), desc="Generating") if show_progress else range(max_len)
            
            # Generate tokens one by one
            for _ in progress_iter:
                # Get predictions for the current sequence
                logits, _ = self.model(x)
                
                # Get the next token predictions (last token in sequence)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    # Get the top-k values and indices
                    values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                    # Create a mask of the top-k positions
                    mask = torch.zeros_like(next_token_logits).scatter_(1, indices, 1.0)
                    # Apply the mask (set non-top-k values to -inf)
                    next_token_logits = torch.where(mask > 0, next_token_logits, torch.tensor(-float('inf')).to(next_token_logits.device))
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.EOS_IDX:
                    break
                
                # Append the new token to the sequence
                x = torch.cat([x, next_token], dim=1)
        
        # Convert indices back to words
        generated_text = []
        for idx in x[0]:
            if idx.item() == self.BOS_IDX:
                continue
            if idx.item() == self.EOS_IDX:
                break
            generated_text.append(self.vocab[idx.item()])
        
        # Join tokens to form the text
        return " ".join(generated_text)
    
    def predict_next_tokens(self, prompt, num_predictions=5):
        """
        Predict the most likely next tokens given a prompt.
        
        Args:
            prompt (str): The prompt text
            num_predictions (int): Number of top predictions to return
            
        Returns:
            list: List of (token, probability) tuples for the top predictions
        """
        self.model.eval()
        
        # Tokenize the prompt
        tokens = re.findall(r'\b\w+\b|[^\w\s]', prompt.strip().lower())
        input_ids = [self.BOS_IDX]
        for token in tokens:
            input_ids.append(self.word2idx.get(token, self.UNK_IDX))
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            
            # Get the predictions for the next token (last position)
            next_token_logits = logits[0, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Get the top k predictions
            topk_probs, topk_indices = torch.topk(probs, k=num_predictions)
            
            # Convert to list of (token, probability) tuples
            predictions = []
            for i in range(num_predictions):
                idx = topk_indices[i].item()
                prob = topk_probs[i].item()
                token = self.vocab[idx]
                predictions.append((token, prob))
            
            return predictions

# =============================================================================
# Interactive Interface
# =============================================================================

def interactive_generation():
    """
    Run an interactive text generation session.
    """
    parser = argparse.ArgumentParser(description="Text generation with a trained language model")
    parser.add_argument("--model_path", type=str, default="models/gpt_best.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--max_len", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Sample from top k most likely tokens")
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first or provide the correct path.")
        return
    
    # Initialize the language model
    lm = LanguageModelInference(args.model_path)
    
    print("\nWelcome to the Language Model Text Generator")
    print("Enter a prompt to generate text, or 'q' to quit")
    print("Commands:")
    print("  /temp X   - Set temperature to X (e.g., /temp 0.5)")
    print("  /topk X   - Set top-k to X (e.g., /topk 20)")
    print("  /len X    - Set max length to X (e.g., /len 100)")
    print("  /predict  - Show top 5 next token predictions")
    print("  /verbose  - Toggle progress bar during generation")
    print("-" * 50)
    
    # Default settings
    temperature = args.temperature
    top_k = args.top_k
    max_len = args.max_len
    show_progress = False
    
    while True:
        prompt = input("\nPrompt: ")
        
        # Check for quit command
        if prompt.lower() in ('q', 'quit', 'exit'):
            break
        
        # Check for commands
        if prompt.startswith('/temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature. Should be a float.")
            continue
            
        elif prompt.startswith('/topk '):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
            except:
                print("Invalid top-k. Should be an integer.")
            continue
            
        elif prompt.startswith('/len '):
            try:
                max_len = int(prompt.split()[1])
                print(f"Max length set to {max_len}")
            except:
                print("Invalid length. Should be an integer.")
            continue
            
        elif prompt.startswith('/verbose'):
            show_progress = not show_progress
            print(f"Progress bar {'enabled' if show_progress else 'disabled'}")
            continue
            
        elif prompt.startswith('/predict'):
            if len(prompt) > 8:  # If there's text after "/predict"
                input_text = prompt[8:].strip()
            else:
                input_text = input("Enter text for prediction: ")
                
            if input_text:
                predictions = lm.predict_next_tokens(input_text)
                print("\nTop 5 predictions for next token:")
                for i, (token, prob) in enumerate(predictions):
                    print(f"  {i+1}. '{token}' (probability: {prob:.4f})")
            else:
                print("No input provided for prediction.")
            continue
        
        # Generate text
        if prompt:
            print("\nGenerating...")
            generated_text = lm.generate_text(
                prompt=prompt,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                show_progress=show_progress
            )
            print(f"\nGenerated: {generated_text}")
        else:
            print("Please enter a prompt.")
    
    print("\nThank you for using the Language Model Text Generator!")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    interactive_generation() 