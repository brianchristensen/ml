"""Train Compositional HRR Model on COGS dataset."""

from model import CompositionalCOGSModel
import time
import sys


def load_cogs(split='train', limit=None):
    """Load COGS dataset."""
    data_dir = 'data/cogs_repo/data'
    filepath = f'{data_dir}/{split}.tsv'

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            sentence = parts[0].strip()
            logical_form = parts[1].strip()

            # Tokenize sentence
            tokens = sentence.lower().replace('.', '').split()

            data.append((tokens, logical_form))

            if limit and len(data) >= limit:
                break

    return data


def train_model(train_limit=None):
    """Train model on COGS dataset."""
    print("="*70)
    print("TRAINING COMPOSITIONAL HRR MODEL")
    print("="*70)
    print()

    # Load data
    train_data = load_cogs('train', limit=train_limit)

    print(f"Training on {len(train_data):,} examples")
    print()

    # Create model
    model = CompositionalCOGSModel(hrr_dim=2048)

    # Train
    start = time.time()
    model.train_on_dataset(train_data)
    train_time = time.time() - start

    print(f"Training completed in {train_time:.2f}s")
    print(f"Examples stored: {len(model.examples):,}")
    print(f"Lexicon size: {len(model.lexicon):,}")
    print()
    print("="*70)

    return model


if __name__ == '__main__':
    # Parse command line args
    train_limit = None
    if len(sys.argv) > 1:
        train_limit = int(sys.argv[1])
        print(f"Training on {train_limit} examples (limited)")
    else:
        print("Training on full dataset")

    print()

    model = train_model(train_limit=train_limit)
