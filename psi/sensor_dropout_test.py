"""
Sensor Dropout Test: Robustness to Sensor Failure

Test whether PSI learned the underlying dynamical system's phase space,
or just memorized the specific 3-sensor pattern.

If PSI learned the true dynamics, it should gracefully degrade when
a sensor fails - using the remaining sensors to still track the object.

This simulates real-world scenarios:
- Lidar blocked by fog/rain
- Radar jammed or failed
- Camera occluded or broken
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sensor_fusion_experiment import (
    SensorFusionDataset,
    PSISensorFusionModel,
    RADAR, LIDAR, CAMERA, IMU,
    SENSOR_INPUT_DIM,
    compute_input_dim,
    evaluate_model,
    device
)

# Sensor indices in synchronized format (now with 4 sensors)
SENSOR_INDICES = {
    'radar': 0,
    'lidar': 1,
    'camera': 2,
    'imu': 3
}

SENSOR_NAMES = ['radar', 'lidar', 'camera', 'imu']

# Track which sensors provide velocity
VELOCITY_SENSORS = ['radar', 'imu']


def dropout_sensor(inputs, sensor_name):
    """
    Zero out all observations from a specific sensor.

    In the synchronized format, each sensor occupies SENSOR_INPUT_DIM (8) values:
    [valid, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, confidence]

    This simulates complete sensor failure by zeroing all values for that sensor
    across all timesteps.
    """
    inputs_modified = inputs.clone()
    sensor_idx = SENSOR_INDICES[sensor_name]
    offset = sensor_idx * SENSOR_INPUT_DIM

    # Count how many valid readings we're dropping
    # valid flag is at offset position for each sensor
    valid_mask = inputs[:, :, offset] == 1.0

    # Zero out all 8 values for this sensor across all timesteps
    inputs_modified[:, :, offset:offset + SENSOR_INPUT_DIM] = 0.0

    return inputs_modified, valid_mask


def evaluate_with_dropout(model, dataloader, sensor_to_drop, device):
    """Evaluate model with one sensor completely dropped."""
    model.eval()

    position_errors = []
    velocity_errors = []
    n_dropped_readings = 0
    n_total_readings = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Drop sensor
            inputs_dropped, mask = dropout_sensor(inputs, sensor_to_drop)
            n_dropped_readings += mask.sum().item()
            n_total_readings += mask.numel()

            # Predict with dropped sensor
            pred = model(inputs_dropped)

            # Compute errors
            pos_err = torch.sqrt(((pred[:, :, :3] - targets[:, :, :3])**2).sum(dim=-1))
            vel_err = torch.sqrt(((pred[:, :, 3:6] - targets[:, :, 3:6])**2).sum(dim=-1))

            position_errors.extend(pos_err.cpu().numpy().flatten())
            velocity_errors.extend(vel_err.cpu().numpy().flatten())

    drop_percentage = n_dropped_readings / n_total_readings * 100

    return {
        'position_rmse': np.sqrt(np.mean(np.array(position_errors)**2)),
        'velocity_rmse': np.sqrt(np.mean(np.array(velocity_errors)**2)),
        'position_mae': np.mean(position_errors),
        'velocity_mae': np.mean(velocity_errors),
        'readings_dropped_pct': drop_percentage
    }


def main():
    parser = argparse.ArgumentParser(description='Sensor Dropout Robustness Test')
    parser.add_argument('--n_trajectories', type=int, default=500,
                        help='Number of test trajectories')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Sequence length')
    parser.add_argument('--dim', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of PSI layers')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    args = parser.parse_args()

    print("=" * 80)
    print("SENSOR DROPOUT ROBUSTNESS TEST")
    print("=" * 80)
    print()
    print("Question: Did PSI learn the underlying dynamics, or just the")
    print("          specific 3-sensor observation pattern?")
    print()
    print("If PSI learned the dynamics, dropping a sensor should cause")
    print("graceful degradation, not catastrophic failure.")
    print("=" * 80)
    print()

    # Load trained model
    print("Loading trained model...")
    sensors = [RADAR, LIDAR, CAMERA, IMU]
    input_dim = compute_input_dim(len(sensors))  # Synchronized format: 4 sensors * 8 values = 32

    model = PSISensorFusionModel(
        input_dim=input_dim,
        state_dim=6,
        hidden_dim=args.dim,
        num_layers=args.num_layers
    ).to(device)

    try:
        checkpoint = torch.load('sensor_fusion_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch with val_loss={checkpoint.get('val_loss', 'unknown')}")
    except FileNotFoundError:
        print("ERROR: sensor_fusion_model.pt not found!")
        print("Please train the sensor fusion model first:")
        print("  python sensor_fusion_experiment.py")
        return

    # Generate test data
    print("\nGenerating test data...")
    test_dataset = SensorFusionDataset(
        n_trajectories=args.n_trajectories,
        duration=10.0,
        sensors=sensors,
        seq_len=args.seq_len,
        split='test',
        seed=999  # Different seed for fresh test data
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Baseline: all sensors working
    print("\n" + "=" * 80)
    print("BASELINE: All Sensors Working")
    print("=" * 80)

    baseline = evaluate_model(model, test_loader, device)
    print(f"  Position RMSE: {baseline['position_rmse']:.3f} m")
    print(f"  Velocity RMSE: {baseline['velocity_rmse']:.3f} m/s")

    # Test each sensor dropout
    results = {'baseline': baseline}

    for sensor_name in SENSOR_NAMES:
        print(f"\n" + "=" * 80)
        print(f"DROPOUT: {sensor_name.upper()} Failed")
        print("=" * 80)

        metrics = evaluate_with_dropout(model, test_loader, sensor_name, device)
        results[f'no_{sensor_name}'] = metrics

        print(f"  Readings dropped: {metrics['readings_dropped_pct']:.1f}%")
        print(f"  Position RMSE: {metrics['position_rmse']:.3f} m")
        print(f"  Velocity RMSE: {metrics['velocity_rmse']:.3f} m/s")

        # Degradation
        pos_degradation = (metrics['position_rmse'] - baseline['position_rmse']) / baseline['position_rmse'] * 100
        vel_degradation = (metrics['velocity_rmse'] - baseline['velocity_rmse']) / baseline['velocity_rmse'] * 100
        print(f"  Position degradation: {pos_degradation:+.1f}%")
        print(f"  Velocity degradation: {vel_degradation:+.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Condition':<20} {'Pos RMSE':>10} {'Vel RMSE':>10} {'Pos Δ':>10} {'Vel Δ':>10}")
    print("-" * 62)

    print(f"{'All sensors':<20} {baseline['position_rmse']:>10.3f} {baseline['velocity_rmse']:>10.3f} {'---':>10} {'---':>10}")

    for sensor_name in SENSOR_NAMES:
        metrics = results[f'no_{sensor_name}']
        pos_deg = (metrics['position_rmse'] - baseline['position_rmse']) / baseline['position_rmse'] * 100
        vel_deg = (metrics['velocity_rmse'] - baseline['velocity_rmse']) / baseline['velocity_rmse'] * 100
        print(f"{'No ' + sensor_name:<20} {metrics['position_rmse']:>10.3f} {metrics['velocity_rmse']:>10.3f} {pos_deg:>+9.1f}% {vel_deg:>+9.1f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conditions = ['All\nsensors'] + [f'No\n{s}' for s in SENSOR_NAMES]
    pos_rmses = [baseline['position_rmse']] + [results[f'no_{s}']['position_rmse'] for s in SENSOR_NAMES]
    vel_rmses = [baseline['velocity_rmse']] + [results[f'no_{s}']['velocity_rmse'] for s in SENSOR_NAMES]

    colors = ['green'] + ['red', 'orange', 'gold']

    axes[0].bar(conditions, pos_rmses, color=colors)
    axes[0].set_ylabel('Position RMSE (m)')
    axes[0].set_title('Position Estimation Under Sensor Failure')
    axes[0].axhline(y=baseline['position_rmse'], color='green', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(conditions, vel_rmses, color=colors)
    axes[1].set_ylabel('Velocity RMSE (m/s)')
    axes[1].set_title('Velocity Estimation Under Sensor Failure')
    axes[1].axhline(y=baseline['velocity_rmse'], color='green', linestyle='--', alpha=0.5, label='Baseline')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('sensor_dropout_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved sensor_dropout_results.png")
    plt.close()

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    avg_pos_degradation = np.mean([
        (results[f'no_{s}']['position_rmse'] - baseline['position_rmse']) / baseline['position_rmse'] * 100
        for s in SENSOR_NAMES
    ])

    if avg_pos_degradation < 50:
        print("\n✓ GRACEFUL DEGRADATION")
        print(f"  Average position degradation: {avg_pos_degradation:.1f}%")
        print("  PSI learned the underlying dynamics and can compensate for sensor loss.")
        print("  It's using redundant information from remaining sensors.")
    elif avg_pos_degradation < 100:
        print("\n~ MODERATE DEGRADATION")
        print(f"  Average position degradation: {avg_pos_degradation:.1f}%")
        print("  PSI partially learned the dynamics but relies significantly on all sensors.")
    else:
        print("\n✗ CATASTROPHIC FAILURE")
        print(f"  Average position degradation: {avg_pos_degradation:.1f}%")
        print("  PSI memorized the 3-sensor pattern rather than learning dynamics.")

    # Which sensor matters most?
    degradations = {s: (results[f'no_{s}']['position_rmse'] - baseline['position_rmse']) / baseline['position_rmse'] * 100
                    for s in SENSOR_NAMES}
    most_important = max(degradations.keys(), key=lambda k: degradations[k])
    least_important = min(degradations.keys(), key=lambda k: degradations[k])

    print(f"\n  Most critical sensor: {most_important.upper()} (dropping causes {degradations[most_important]:.1f}% degradation)")
    print(f"  Least critical sensor: {least_important.upper()} (dropping causes {degradations[least_important]:.1f}% degradation)")


if __name__ == "__main__":
    main()
