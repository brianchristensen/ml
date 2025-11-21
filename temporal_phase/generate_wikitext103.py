"""
Text Generation from Trained WikiText-103 Model

Usage:
    python generate_wikitext103.py
    python generate_wikitext103.py --checkpoint wikitext103_best.pt --prompt "The history of"
"""

import torch
import torch.nn.functional as F
import tiktoken
import argparse
from pathlib import Path

from novel_attention import NovelAttentionLM


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Get tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create model
    model = NovelAttentionLM(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        num_layers=config['num_layers'],
        max_len=config['seq_len'],
        device=device
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    print(f"  Validation PPL: {checkpoint['val_ppl']:.2f}")
    print()

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50, device='cuda'):
    """
    Generate text from a prompt.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Text prompt to continue
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = greedy)
        device: Device

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=len(tokens) + max_length,
            temperature=temperature,
            top_k=top_k
        )

    # Decode
    generated_tokens = generated[0].cpu().numpy().tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def interactive_generation(model, tokenizer, device='cuda'):
    """Interactive generation loop."""
    print("=" * 80)
    print("Interactive Text Generation")
    print("=" * 80)
    print()
    print("Commands:")
    print("  Type a prompt and press Enter to generate")
    print("  'temp X' to set temperature (e.g., 'temp 0.8')")
    print("  'len X' to set max length (e.g., 'len 100')")
    print("  'topk X' to set top-k (e.g., 'topk 50', 'topk 0' for greedy)")
    print("  'quit' or 'exit' to quit")
    print()

    # Default settings
    temperature = 0.8
    max_length = 150
    top_k = 50

    while True:
        print("-" * 80)
        print(f"Settings: temp={temperature}, max_length={max_length}, top_k={top_k}")
        prompt = input("\nPrompt: ").strip()

        if not prompt:
            continue

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Check for commands
        if prompt.startswith('temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid temperature. Use: temp 0.8")
                continue

        if prompt.startswith('len '):
            try:
                max_length = int(prompt.split()[1])
                print(f"Max length set to {max_length}")
                continue
            except:
                print("Invalid length. Use: len 100")
                continue

        if prompt.startswith('topk '):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
                continue
            except:
                print("Invalid top-k. Use: topk 50")
                continue

        # Generate
        print()
        print("=" * 80)
        print("GENERATED:")
        print("=" * 80)

        try:
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            print(generated)
        except Exception as e:
            print(f"Generation failed: {e}")

        print()


def batch_generation(model, tokenizer, prompts, output_file=None,
                     max_length=200, temperature=0.8, top_k=50, device='cuda'):
    """
    Generate text for multiple prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        output_file: Optional file to save results
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device
    """
    print("=" * 80)
    print("Batch Generation")
    print("=" * 80)
    print()

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating from: {prompt[:50]}...")

        generated = generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            device=device
        )

        results.append({
            'prompt': prompt,
            'generated': generated
        })

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"[{i}] Prompt: {result['prompt']}")
        print(f"{'='*80}")
        print(result['generated'])
        print()

    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"{'='*80}\n")
                f.write(f"[{i}] Prompt: {result['prompt']}\n")
                f.write(f"{'='*80}\n")
                f.write(result['generated'])
                f.write("\n\n")

        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained WikiText-103 model")
    parser.add_argument('--checkpoint', type=str, default='wikitext103_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt (if not provided, enters interactive mode)')
    parser.add_argument('--prompts-file', type=str, default=None,
                       help='File with prompts (one per line) for batch generation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for batch generation')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling (0 for greedy)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print()
        print("Train a model first with: python train_wikitext103.py")
        return

    # Load model
    model, tokenizer = load_model(args.checkpoint, device=args.device)

    # Batch generation from file
    if args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        batch_generation(
            model, tokenizer, prompts,
            output_file=args.output,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )

    # Single prompt generation
    elif args.prompt:
        print("=" * 80)
        print("PROMPT:")
        print("=" * 80)
        print(args.prompt)
        print()
        print("=" * 80)
        print("GENERATED:")
        print("=" * 80)

        generated = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )

        print(generated)
        print()

    # Interactive mode
    else:
        interactive_generation(model, tokenizer, device=args.device)


if __name__ == "__main__":
    main()
