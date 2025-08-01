import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import sys

from gpt import GPTLanguageModel

torch.manual_seed(43)

def load_model(model_path):
    """
    Load a trained GPT model from a checkpoint file.
    
    Args:
        model_path (str): Path to the .pth checkpoint file containing the saved model
    
    Returns:
        tuple: A tuple containing:
            - model (GPTLanguageModel): The loaded and initialized model
            - decode (function): Function to decode token indices to text
            - stoi (dict): String to index mapping dictionary
            - itos (dict): Index to string mapping dictionary
    """
    # Load saved checkpoint data
    checkpoint = torch.load(model_path)
    
    # Restore hyperparameters from checkpoint
    vocab_size = checkpoint['vocab_size']
    n_embd = checkpoint['n_embd']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    block_size = checkpoint['block_size']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Rebuild model structure (requires GPTLanguageModel class definition)
    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create decode function to convert token indices back to text
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return model, decode, stoi, itos

def main():
    """
    Main function to handle command-line arguments and run text generation.

    Example usage:
        python code/load_model.py result/gpt_model_least_val_loss.pth --max_tokens 100
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Load GPT model and generate text')
    parser.add_argument('model_path', 
                       help='Path to the .pth model file (e.g., gpt_model.pth)')
    parser.add_argument('--max_tokens', 
                       type=int, 
                       default=1000, 
                       help='Maximum number of tokens to generate (default: 1000)')
    
    args = parser.parse_args()
    
    try:
        # Use saved model to generate text
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print(f"Loading model from: {args.model_path}")
        
        # Load the model and related components
        model, decode, stoi, itos = load_model(args.model_path)
        model = model.to(device)
        
        # Create encode function to convert text to token indices
        encode = lambda s: [stoi[c] for c in s]
        
        # Start generation from "Hi" instead of empty context
        start_text = "Hi, I love the challenge of learning new things. What abo"
        context = torch.tensor(encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
        print(f"Starting generation from: '{start_text}'")
        print(f"Generating {args.max_tokens} additional tokens...")
        
        generated_tokens = model.generate(context, max_new_tokens=args.max_tokens)[0].tolist()
        generated_text = decode(generated_tokens)
        
        # Display generated text
        print("\nGenerated text:")
        print("=" * 50)
        print(generated_text)
        
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()