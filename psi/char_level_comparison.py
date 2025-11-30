"""
Character-Level Language Modeling: PSI vs LSTM vs Transformer

HISTORICAL CONTEXT:
- char-rnn (Karpathy 2015) showed LSTMs can learn language from characters
- Transformers replaced LSTMs but mostly use BPE tokenization
- Character-level is harder: model must learn spelling, morphology, syntax

THE HYPOTHESIS:
PSI combines LSTM's cumulative state with Transformer's parallelism.
If true, PSI should:
1. Match LSTM on character-level language (cumulative state)
2. Beat LSTM on training speed (parallel cumsum)
3. Beat Transformer on character-level (cumulative vs attention)

TESTS:
1. Character-level perplexity on text
2. Training speed comparison
3. Generation quality
4. Long sequence handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        cumsum_v = torch.cumsum(g * v, dim=1)
        cumsum_g = torch.cumsum(g, dim=1) + 1e-6
        mem = cumsum_v / cumsum_g
        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIModel(nn.Module):
    def __init__(self, vocab_size=256, dim=256, num_layers=6, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=256, dim=256, num_layers=6, num_heads=8, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=256, dim=256, num_layers=6, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        h, _ = self.lstm(h)
        return self.head(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Dataset
# =============================================================================

# Sample text for testing (Shakespeare-style, public domain)
SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action. Soft you now!
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.
""" * 100  # Repeat to get more data


class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        # Convert to bytes (0-255)
        self.data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
        self.n_seqs = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return {
            'input': chunk[:-1],
            'target': chunk[1:]
        }


def collate_fn(batch):
    return {
        'input': torch.stack([b['input'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch])
    }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    return total_loss / total_tokens


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)  # bits per character

    return ppl, bpc


def generate(model, prompt, max_len=100, temperature=0.8):
    model.eval()
    tokens = list(prompt.encode('utf-8'))
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

            # Stop at newline after some characters
            if next_token == ord('\n') and input_ids.shape[1] > len(tokens) + 20:
                break

    return bytes(input_ids[0].cpu().tolist()).decode('utf-8', errors='replace')


def measure_training_speed(model, loader, n_batches=50):
    """Measure training throughput."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_tokens += targets.numel()

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = total_tokens / elapsed

    return tokens_per_sec


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("CHARACTER-LEVEL LANGUAGE MODELING: PSI vs LSTM vs Transformer")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Config
    dim = 256
    num_layers = 6
    seq_len = 128
    batch_size = 32
    epochs = 20
    vocab_size = 256  # byte-level

    # Create datasets
    n_chars = len(SAMPLE_TEXT.encode('utf-8'))
    train_text = SAMPLE_TEXT[:int(n_chars * 0.9)]
    val_text = SAMPLE_TEXT[int(n_chars * 0.9):]

    train_data = CharDataset(train_text, seq_len=seq_len)
    val_data = CharDataset(val_text, seq_len=seq_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Training data: {len(train_text):,} characters, {len(train_data)} sequences")
    print(f"Validation data: {len(val_text):,} characters, {len(val_data)} sequences")
    print(f"Sequence length: {seq_len}")
    print()

    # Models to test
    models = {
        'PSI': lambda: PSIModel(vocab_size, dim, num_layers, max_len=seq_len+10),
        'Transformer': lambda: TransformerModel(vocab_size, dim, num_layers, max_len=seq_len+10),
        'LSTM': lambda: LSTMModel(vocab_size, dim, num_layers, max_len=seq_len+10),
    }

    results = {}

    for name, model_fn in models.items():
        print("=" * 70)
        print(f"Training {name}")
        print("=" * 70)

        model = model_fn().to(device)
        n_params = model.count_parameters()
        print(f"Parameters: {n_params:,}")

        # Measure training speed first
        print("Measuring training speed...")
        tokens_per_sec = measure_training_speed(model, train_loader)
        print(f"Training speed: {tokens_per_sec:,.0f} tokens/sec")

        # Reset model for actual training
        model = model_fn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

        print(f"\nTraining for {epochs} epochs...")
        train_start = time.time()

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_ppl, val_bpc = evaluate(model, val_loader)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_ppl={val_ppl:.2f}, val_bpc={val_bpc:.3f}")

        train_time = time.time() - train_start

        # Final evaluation
        val_ppl, val_bpc = evaluate(model, val_loader)
        print(f"\nFinal: val_ppl={val_ppl:.2f}, val_bpc={val_bpc:.3f}")
        print(f"Training time: {train_time:.1f}s")

        # Generate sample
        print("\nGeneration sample:")
        prompt = "To be, or not"
        generated = generate(model, prompt, max_len=150, temperature=0.8)
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: {generated[:200]}")

        results[name] = {
            'params': n_params,
            'val_ppl': val_ppl,
            'val_bpc': val_bpc,
            'train_time': train_time,
            'tokens_per_sec': tokens_per_sec,
            'model': model
        }

        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Character-Level Language Modeling")
    print("=" * 80)
    print()

    print(f"{'Model':<15} {'Params':>12} {'Val PPL':>10} {'Val BPC':>10} {'Tok/sec':>12} {'Time':>10}")
    print("-" * 75)

    for name, r in results.items():
        print(f"{name:<15} {r['params']:>12,} {r['val_ppl']:>10.2f} {r['val_bpc']:>10.3f} {r['tokens_per_sec']:>12,.0f} {r['train_time']:>9.1f}s")

    print()

    # Speed comparison
    psi_speed = results['PSI']['tokens_per_sec']
    lstm_speed = results['LSTM']['tokens_per_sec']
    trans_speed = results['Transformer']['tokens_per_sec']

    print("Speed comparison:")
    print(f"  PSI vs LSTM: {psi_speed/lstm_speed:.2f}x")
    print(f"  PSI vs Transformer: {psi_speed/trans_speed:.2f}x")
    print(f"  Transformer vs LSTM: {trans_speed/lstm_speed:.2f}x")

    print()

    # Quality comparison
    psi_bpc = results['PSI']['val_bpc']
    lstm_bpc = results['LSTM']['val_bpc']
    trans_bpc = results['Transformer']['val_bpc']

    print("Quality comparison (lower BPC = better):")
    print(f"  PSI: {psi_bpc:.3f} BPC")
    print(f"  LSTM: {lstm_bpc:.3f} BPC")
    print(f"  Transformer: {trans_bpc:.3f} BPC")

    # =========================================================================
    # Test on longer sequences
    # =========================================================================
    print()
    print("=" * 80)
    print("LONGER SEQUENCE TEST")
    print("=" * 80)

    for test_seq_len in [256, 512]:
        print(f"\nTesting at seq_len={test_seq_len}...")

        test_data = CharDataset(val_text, seq_len=test_seq_len)
        if len(test_data) < 5:
            print(f"  Not enough data for seq_len={test_seq_len}")
            continue

        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

        for name in ['PSI', 'LSTM', 'Transformer']:
            model = results[name]['model']

            # Need to extend position embeddings for longer sequences
            if hasattr(model, 'pos_embed') and model.pos_embed.shape[1] < test_seq_len:
                print(f"  {name}: Position embeddings too short, skipping")
                continue

            try:
                ppl, bpc = evaluate(model, test_loader)
                print(f"  {name}: PPL={ppl:.2f}, BPC={bpc:.3f}")
            except Exception as e:
                print(f"  {name}: Failed - {str(e)[:50]}")

    # =========================================================================
    # Key Insight
    # =========================================================================
    print()
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print(f"""
    Character-level language modeling results:

    QUALITY (BPC - lower is better):
    - PSI: {psi_bpc:.3f}
    - LSTM: {lstm_bpc:.3f}
    - Transformer: {trans_bpc:.3f}

    SPEED (tokens/sec - higher is better):
    - PSI: {psi_speed:,.0f}
    - LSTM: {lstm_speed:,.0f}
    - Transformer: {trans_speed:,.0f}

    PSI's advantage:
    1. Cumulative state like LSTM (good for character-level)
    2. Parallelizable like Transformer (fast training)
    3. O(n) memory (scales to long sequences)

    If PSI matches LSTM quality but with Transformer speed,
    it combines the best of both worlds for sequential tasks.
    """)


if __name__ == "__main__":
    main()
