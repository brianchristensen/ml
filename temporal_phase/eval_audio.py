"""
Eval script for audio model (single-step prediction)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from train_audio_waveform import TPIAudioPredictor, AudioWaveformDataset

def plot_waveform_comparison(original, prediction, sample_rate, save_path):
    """Plot original vs predicted waveforms"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    time_orig = np.arange(len(original)) / sample_rate
    time_pred = np.arange(len(prediction)) / sample_rate

    axes[0].plot(time_orig, original, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Original Audio', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_pred, prediction, 'r-', alpha=0.7, linewidth=0.5)
    axes[1].set_title('TPI Prediction (Single-Step)', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    overlay_length = min(len(original), len(prediction), sample_rate * 2)
    time_overlay = np.arange(overlay_length) / sample_rate
    axes[2].plot(time_overlay, original[:overlay_length], 'b-', alpha=0.7, linewidth=0.8, label='Original')
    axes[2].plot(time_overlay, prediction[:overlay_length], 'r--', alpha=0.7, linewidth=0.8, label='Prediction')
    axes[2].set_title('Overlay (First 2 seconds)', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")

def plot_spectrogram_comparison(original, prediction, sample_rate, save_path):
    """Plot spectrograms"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    n_fft = 512
    hop_length = 256

    def compute_spectrogram(audio):
        spec = np.abs(np.fft.rfft(audio.reshape(-1, n_fft), axis=1))
        return 20 * np.log10(spec + 1e-8)

    def frames(audio, frame_size, hop):
        num_frames = (len(audio) - frame_size) // hop + 1
        return np.array([audio[i*hop:i*hop+frame_size] for i in range(num_frames)])

    orig_frames = frames(original, n_fft, hop_length)
    pred_frames = frames(prediction, n_fft, hop_length)

    spec_orig = compute_spectrogram(orig_frames).T
    spec_pred = compute_spectrogram(pred_frames).T

    vmin, vmax = -60, 0
    im0 = axes[0].imshow(spec_orig, aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, len(original)/sample_rate, 0, sample_rate/2],
                         vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Spectrogram', fontsize=12)
    axes[0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im0, ax=axes[0], label='dB')

    im1 = axes[1].imshow(spec_pred, aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, len(prediction)/sample_rate, 0, sample_rate/2],
                         vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted Spectrogram (Single-Step)', fontsize=12)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[1], label='dB')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_rate = 16000
    context_length = 2048

    print("=" * 60)
    print("TPI Audio Generation - Fixed (Single-Step)")
    print("=" * 60)
    print(f"Device: {device}")

    # Load model
    print("\nLoading trained model...")
    model = TPIAudioPredictor(dim=128, num_layers=8).to(device)
    model.load_state_dict(torch.load('best_audio_model.pt', map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Create test sample
    print("\nGenerating test audio...")
    test_dataset = AudioWaveformDataset(
        sample_rate=sample_rate,
        context_length=context_length,
        hop_length=256,
        synthetic=True,
        num_samples=1
    )

    test_waveform = test_dataset.waveforms[0]
    context_samples = test_waveform[:context_length]
    target_samples = test_waveform[context_length:context_length + 8000]  # 0.5 second (faster)

    print(f"Context: {len(context_samples)} samples (~{len(context_samples)/sample_rate*1000:.1f}ms)")
    print(f"Generating: {len(target_samples)} samples (0.5 second)")
    print("NOTE: Single-step generation is slower but maintains phase coherence")

    # Generate
    context_tensor = torch.from_numpy(context_samples).unsqueeze(0).to(device)

    print("\nGenerating audio sample-by-sample...")
    generated = model.generate_continuation(context_tensor, len(target_samples), device)
    generated_audio = generated[0].cpu().numpy()
    print("✓ Generation complete")

    # Metrics
    mse = np.mean((generated_audio - target_samples)**2)
    mae = np.mean(np.abs(generated_audio - target_samples))
    signal_power = np.mean(target_samples**2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

    print(f"\n{'Evaluation Metrics':-^60}")
    print(f"MSE:        {mse:.6f}")
    print(f"MAE:        {mae:.6f}")
    print(f"SNR:        {snr:.2f} dB")
    print(f"{'':-^60}")

    # Save audio
    print("\nSaving audio files...")

    original_audio = test_waveform[:context_length + len(target_samples)]
    original_audio_int16 = np.clip(original_audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write('audio_original_fixed.wav', sample_rate, original_audio_int16)
    print("✓ Saved audio_original_fixed.wav")

    full_generated = np.concatenate([context_samples, generated_audio])
    full_generated_int16 = np.clip(full_generated * 32767, -32768, 32767).astype(np.int16)
    wavfile.write('audio_generated_fixed.wav', sample_rate, full_generated_int16)
    print("✓ Saved audio_generated_fixed.wav")

    generated_only_int16 = np.clip(generated_audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write('audio_prediction_only_fixed.wav', sample_rate, generated_only_int16)
    print("✓ Saved audio_prediction_only_fixed.wav")

    # Visualizations
    print("\nCreating visualizations...")
    plot_waveform_comparison(original_audio, full_generated, sample_rate,
                            'audio_waveform_comparison_fixed.png')
    plot_spectrogram_comparison(original_audio, full_generated, sample_rate,
                               'audio_spectrogram_comparison_fixed.png')

    # Prediction overlay
    fig, ax = plt.subplots(figsize=(15, 5))
    time = np.arange(len(generated_audio)) / sample_rate
    ax.plot(time, target_samples, 'b-', alpha=0.7, linewidth=0.8, label='Ground Truth')
    ax.plot(time, generated_audio, 'r--', alpha=0.7, linewidth=0.8, label='TPI (Single-Step)')
    ax.set_title('0.5s Prediction: Ground Truth vs TPI', fontsize=14)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('audio_prediction_overlay_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved audio_prediction_overlay_fixed.png")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("\nGenerated files:")
    print("  - audio_original_fixed.wav")
    print("  - audio_generated_fixed.wav")
    print("  - audio_prediction_only_fixed.wav")
    print("  - audio_waveform_comparison_fixed.png")
    print("  - audio_spectrogram_comparison_fixed.png")
    print("  - audio_prediction_overlay_fixed.png")
    print("\nListen to the audio - oscillations should be maintained!")
    print("=" * 60)

if __name__ == '__main__':
    main()
