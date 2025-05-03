import os
import torch
import argparse
import tiktoken
from models.model_tauon import GPT, GPTConfig

def get_args():
    parser = argparse.ArgumentParser(description='Generate text using a trained GPT model')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='optional text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='top-k sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='tokenizer to use')
    return parser.parse_args()

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config from the model weights
    state_dict = checkpoint['model_state_dict']
    
    # Try to infer configuration from state_dict
    n_embd = state_dict['transformer.wte.weight'].shape[1]
    vocab_size = state_dict['transformer.wte.weight'].shape[0]
    block_size = state_dict['transformer.wpe.weight'].shape[0]
    n_layer = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith('transformer.h.')]) + 1
    n_head = n_embd // 64  # Just a guess, assuming head size is 64
    
    # Determine rank from L parameters
    first_L_param = next(k for k in state_dict.keys() if '.L' in k)
    rank = state_dict[first_L_param].shape[1]
    
    # Create config and model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        rank=rank
    )
    
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Configuration: {n_layer} layers, {n_head} heads, {n_embd} embedding dim, rank {rank}")
    return model

def main():
    args = get_args()
    
    # Load tokenizer
    if args.tokenizer == 'gpt2':
        print("Using GPT-2 tokenizer")
        tokenizer = tiktoken.get_encoding("gpt2")
    else:
        raise ValueError(f"Tokenizer {args.tokenizer} not supported")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.device)
    
    # Set up context
    if args.prompt:
        print(f"Using prompt: {args.prompt}")
        context_tokens = tokenizer.encode(args.prompt)
        context = torch.tensor(context_tokens, dtype=torch.long, device=args.device).unsqueeze(0)
    else:
        # Generate from a random token
        context = torch.randint(0, model.config.vocab_size, (1, 1), device=args.device)
    
    # Generate text
    print(f"\nGenerating {args.max_new_tokens} tokens with temperature {args.temperature}...\n")
    
    start_len = context.size(1)
    with torch.no_grad():
        output = model.generate(
            context, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    # Decode the output
    generated_tokens = output[0].tolist()
    prompt_tokens = generated_tokens[:start_len]
    new_tokens = generated_tokens[start_len:]
    
    prompt_text = tokenizer.decode(prompt_tokens)
    generated_text = tokenizer.decode(new_tokens)
    
    # Print the result
    if args.prompt:
        print(f"Context: {prompt_text}")
    print(f"\nGenerated text:\n{prompt_text}{generated_text}")
    
    # Save the generated text to a file
    output_dir = os.path.dirname(args.model_path)
    output_file = os.path.join(output_dir, 'generated_text.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{prompt_text}{generated_text}")
    
    print(f"\nGenerated text saved to {output_file}")

if __name__ == '__main__':
    main() 