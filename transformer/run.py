"""
Run script for the transformer language model.
This script allows you to:
1. Train the language model
2. Load a trained model and perform text generation
3. Do both
"""

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run the transformer language model")
    parser.add_argument("--mode", choices=["train", "inference", "both"], 
                        default="both", help="Mode to run the script in")
    parser.add_argument("--model_path", type=str, default="models/gpt_best.pt",
                        help="Path to load/save the model")
    args = parser.parse_args()
    
    # Check if the required packages are installed
    try:
        import torch
        import numpy
        import tqdm
    except ImportError:
        print("Required packages are not installed.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Run based on the selected mode
    if args.mode in ["train", "both"]:
        print("\n=== Training the Language Model ===\n")
        subprocess.run([sys.executable, "train_lm.py"])
    
    if args.mode in ["inference", "both"]:
        print("\n=== Running Text Generation with the Language Model ===\n")
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}")
            if args.mode == "inference":
                print("Please train the model first or specify a valid model path.")
                sys.exit(1)
            else:
                # If we just trained the model, try the final model
                if os.path.exists("models/gpt_final.pt"):
                    args.model_path = "models/gpt_final.pt"
                    print(f"Using the final model at {args.model_path}")
                else:
                    print("No trained model found. Please check if training completed successfully.")
                    sys.exit(1)
        
        subprocess.run([sys.executable, "inference_lm.py", "--model_path", args.model_path])


if __name__ == "__main__":
    main() 