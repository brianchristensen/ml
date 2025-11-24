"""
Train TPI on audio waveform prediction - FIXED VERSION
Single-step prediction to maintain phase coherence
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torchaudio
import torchaudio.transforms as T
from scipy.io import wavfile
from novel_attention import TPIBlock

class AudioWaveformDataset(Dataset):
    """Dataset for audio waveform prediction"""
    def __init__(self, audio_files=None, sample_rate=16000, context_length=2048,
                 hop_length=512, synthetic=True, num_samples=1000):
        self.sample_rate = sample_rate
        self.context_length = context_length
        self.hop_length = hop_length

        if synthetic:
            self.waveforms = self.generate_synthetic_audio(num_samples)
        else:
            self.waveforms = self.load_audio_files(audio_files)

        self.chunks = self.create_chunks()

    def generate_synthetic_audio(self, num_samples):
        """Generate synthetic audio with complex temporal dynamics"""
        waveforms = []
        duration = 4.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        for i in range(num_samples):
            # FM synthesis with harmonics
            f_base = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)
            signal = np.zeros_like(t)
            for harmonic in range(1, 6):
                freq = f_base * harmonic
                amp = np.exp(-0.3 * harmonic) * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t * harmonic))
                signal += amp * np.sin(2 * np.pi * freq * t / self.sample_rate)
            signal += 0.05 * np.random.randn(len(t))
            signal = signal / (np.abs(signal).max() + 1e-8)
            waveforms.append(signal.astype(np.float32))
        return waveforms

    def load_audio_files(self, audio_files):
        """Load audio from files"""
        waveforms = []
        resampler = T.Resample(orig_freq=None, new_freq=self.sample_rate)
        for audio_path in audio_files:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler.orig_freq = sr
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform[0]
            waveforms.append(waveform.numpy())
        return waveforms

    def create_chunks(self):
        """Create chunks - now predicting just 1 sample ahead"""
        chunks = []
        for waveform in self.waveforms:
            for start_idx in range(0, len(waveform) - self.context_length - 1, self.hop_length):
                context = waveform[start_idx:start_idx + self.context_length]
                target = waveform[start_idx + self.context_length]  # Just 1 sample!
                chunks.append({
                    'context': context,
                    'target': target
                })
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return (
            torch.from_numpy(chunk['context']).float(),
            torch.tensor(chunk['target']).float()
        )

class TPIAudioPredictor(nn.Module):
    """TPI model for audio waveform prediction - single-step output"""
    def __init__(self, dim=128, num_layers=8):
        super().__init__()
        self.dim = dim

        # Embed raw waveform samples
        self.waveform_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # TPI blocks
        self.blocks = nn.ModuleList([
            TPIBlock(dim=dim) for _ in range(num_layers)
        ])

        # Output head - predict just 1 sample
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1)  # Single sample output
        )

    def forward(self, waveform):
        """
        Args:
            waveform: [batch_size, context_length] raw audio samples
        Returns:
            prediction: [batch_size, 1] next audio sample
        """
        x = waveform.unsqueeze(-1)  # [B, T, 1]
        x = self.waveform_embedding(x)  # [B, T, dim]

        # Apply TPI blocks
        for block in self.blocks:
            x = block(x)

        # Take final state and predict next sample
        final_state = x[:, -1, :]  # [B, dim]
        prediction = self.output_head(final_state)  # [B, 1]

        return prediction.squeeze(-1)  # [B]

    def generate_continuation(self, context, num_samples, device='cuda'):
        """Generate audio continuation sample-by-sample"""
        self.eval()
        generated = []

        with torch.no_grad():
            current_context = context.clone()

            for step in range(num_samples):
                # Predict next single sample
                next_sample = self.forward(current_context)  # [B]
                generated.append(next_sample)

                # Slide window: drop first sample, append new one
                current_context = torch.cat([
                    current_context[:, 1:],
                    next_sample.unsqueeze(-1)
                ], dim=1)

                if step % 1000 == 0:
                    print(f"  Generated {step}/{num_samples} samples...")

        return torch.stack(generated, dim=1)  # [B, num_samples]

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (context, target) in enumerate(dataloader):
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        prediction = model(context)

        loss = nn.MSELoss()(prediction, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)

            prediction = model(context)
            loss = nn.MSELoss()(prediction, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_rate = 16000
    context_length = 2048
    batch_size = 32
    num_epochs = 2  # Fewer epochs since we have more samples now

    print("TPI Audio Waveform Prediction - FIXED (Single-Step)")
    print("=" * 60)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Context: {context_length} samples (~{context_length/sample_rate*1000:.1f}ms)")
    print(f"Prediction: 1 sample at a time (phase-coherent)")
    print(f"Device: {device}")

    # Create dataset
    print("\nGenerating synthetic audio dataset...")
    train_dataset = AudioWaveformDataset(
        sample_rate=sample_rate,
        context_length=context_length,
        hop_length=256,
        synthetic=True,
        num_samples=500
    )

    val_dataset = AudioWaveformDataset(
        sample_rate=sample_rate,
        context_length=context_length,
        hop_length=256,
        synthetic=True,
        num_samples=50
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("\nInitializing TPI Audio Predictor...")
    model = TPIAudioPredictor(
        dim=128,
        num_layers=8
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_audio_model.pt')
            print(f"✓ Saved best model (val loss: {val_loss:.6f})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("\nRun eval_audio.py to generate audio samples")

if __name__ == '__main__':
    main()
