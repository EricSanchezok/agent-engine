#!/usr/bin/env python3
"""
Compare risk labels before and after dilation-erosion operations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ICURiskPredictionAgent.icu_sequence_dataloader import ICUSequenceDataLoader
from agents.ICURiskPredictionAgent.risk_label_indexer import RiskLabelIndexer

def convert_timestamps_to_datetime(timestamps: List[str]) -> List[datetime]:
    """Convert timestamp strings to datetime objects."""
    datetime_objects = []
    for timestamp in timestamps:
        try:
            # Parse timestamp string
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            datetime_objects.append(dt)
        except ValueError:
            # If parsing fails, use a default datetime
            datetime_objects.append(datetime.now())
    return datetime_objects

def calculate_cumulative_hours(datetime_objects: List[datetime]) -> List[float]:
    """Calculate cumulative hours from the first event."""
    if not datetime_objects:
        return []
    
    first_event = datetime_objects[0]
    cumulative_hours = []
    
    for dt in datetime_objects:
        time_diff = dt - first_event
        hours = time_diff.total_seconds() / 3600.0
        cumulative_hours.append(hours)
    
    return cumulative_hours

def plot_comparison(
    patient_id: str,
    data_dir: str = "database/icu_patients",
    max_risks: int = 5,
    figsize: tuple = (20, 12),
    # Risk smoothing hyperparameters
    dilation_window_hours: float = 48.0,
    erosion_window_hours: float = 24.0,
    risk_growth_rate: float = 2.0,
    risk_decay_rate: float = 4.0,
):
    """
    Plot comparison between original and dilation-erosion processed risk labels.
    
    Args:
        patient_id: Patient ID to visualize
        data_dir: Directory containing patient data
        max_risks: Maximum number of risks to display
        figsize: Figure size for the plot
        dilation_window_hours: Time window for dilation operation
        erosion_window_hours: Time window for erosion operation
        risk_growth_rate: Exponential rate for risk appearance
        risk_decay_rate: Exponential rate for risk disappearance
    """
    print(f"Comparing dilation-erosion effects for patient: {patient_id}")
    
    # Load data with dilation-erosion
    print("Loading data with dilation-erosion...")
    loader_with_de = ICUSequenceDataLoader(
        data_dir=data_dir,
        only_patient_id=patient_id,
        train_ratio=0.6,
        shuffle_patients=False,
        dilation_window_hours=dilation_window_hours,
        erosion_window_hours=erosion_window_hours,
        risk_growth_rate=risk_growth_rate,
        risk_decay_rate=risk_decay_rate,
    )
    
    # Load data without dilation-erosion (set window to 0)
    print("Loading data without dilation-erosion...")
    loader_without_de = ICUSequenceDataLoader(
        data_dir=data_dir,
        only_patient_id=patient_id,
        train_ratio=0.6,
        shuffle_patients=False,
        dilation_window_hours=0.0,  # No dilation
        erosion_window_hours=0.0,   # No erosion
        risk_growth_rate=risk_growth_rate,
        risk_decay_rate=risk_decay_rate,
    )
    
    # Get risk indexer for risk names
    indexer = RiskLabelIndexer()
    
    # Load patient data
    patient_data_with_de = None
    patient_data_without_de = None
    
    for sample in loader_with_de.iter_patient_sequences("train"):
        patient_data_with_de = sample
        break
    
    for sample in loader_without_de.iter_patient_sequences("train"):
        patient_data_without_de = sample
        break
    
    if patient_data_with_de is None or patient_data_without_de is None:
        print(f"No data found for patient {patient_id}")
        return
    
    # Extract data
    timestamps = patient_data_with_de["timestamps"]
    Y_original = patient_data_with_de["labels"]
    Y_with_de = patient_data_with_de["labels_smoothed"]
    Y_without_de = patient_data_without_de["labels_smoothed"]
    
    # Convert timestamps to cumulative hours
    datetime_objects = convert_timestamps_to_datetime(timestamps)
    cumulative_hours = calculate_cumulative_hours(datetime_objects)
    
    print(f"Data loaded: {len(timestamps)} events, {Y_original.shape[1]} risks")
    print(f"Time range: {cumulative_hours[0]:.2f} to {cumulative_hours[-1]:.2f} hours")
    print(f"Total duration: {cumulative_hours[-1]:.2f} hours ({cumulative_hours[-1]/24:.2f} days)")
    
    # Find risks with occurrences
    risk_occurrences = []
    for risk_idx in range(Y_original.shape[1]):
        original_count = np.count_nonzero(Y_original[:, risk_idx])
        if original_count > 0:
            risk_occurrences.append((risk_idx, original_count))
    
    # Sort by occurrence count and take top risks
    risk_occurrences.sort(key=lambda x: x[1], reverse=True)
    top_risks = risk_occurrences[:max_risks]
    
    print(f"Found {len(risk_occurrences)} risks with occurrences")
    print(f"Displaying top {len(top_risks)} most frequent risks")
    
    # Create comparison plot with 3 subplots: Original, Without DE, With DE
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Define colors for different risks
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot 1: Original data
    ax1 = axes[0]
    for i, (risk_idx, original_count) in enumerate(top_risks):
        risk_name = indexer.index_to_risk(risk_idx)
        color = colors[i % len(colors)]
        ax1.plot(cumulative_hours, Y_original[:, risk_idx], color=color, alpha=0.8, linewidth=1.5, 
                label=f'Risk {risk_idx}: {risk_name}')
    
    ax1.set_ylabel('Risk Value')
    ax1.set_title('Original Risk Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Without dilation-erosion
    ax2 = axes[1]
    for i, (risk_idx, original_count) in enumerate(top_risks):
        risk_name = indexer.index_to_risk(risk_idx)
        color = colors[i % len(colors)]
        ax2.plot(cumulative_hours, Y_without_de[:, risk_idx], color=color, alpha=0.8, linewidth=1.5, 
                label=f'Risk {risk_idx}: {risk_name}')
    
    ax2.set_ylabel('Risk Value')
    ax2.set_title('Without Dilation-Erosion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: With dilation-erosion
    ax3 = axes[2]
    for i, (risk_idx, original_count) in enumerate(top_risks):
        risk_name = indexer.index_to_risk(risk_idx)
        color = colors[i % len(colors)]
        ax3.plot(cumulative_hours, Y_with_de[:, risk_idx], color=color, alpha=0.8, linewidth=1.5, 
                label=f'Risk {risk_idx}: {risk_name}')
    
    ax3.set_ylabel('Risk Value')
    ax3.set_title('With Dilation-Erosion')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Set x-axis label
    ax3.set_xlabel('Time (Hours from Admission)')
    
    # Add overall title
    fig.suptitle(f'Dilation-Erosion Comparison for Patient {patient_id}\n'
                f'Dilation: {dilation_window_hours}h, Erosion: {erosion_window_hours}h, Growth Rate: {risk_growth_rate}, Decay Rate: {risk_decay_rate}', 
                fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary for Patient {patient_id} ===")
    print(f"Total events: {len(timestamps)}")
    print(f"Active risks: {len(top_risks)}")
    print(f"Original non-zero values: {np.count_nonzero(Y_original)}")
    print(f"Without dilation-erosion: {np.count_nonzero(Y_without_de)}")
    print(f"With dilation-erosion: {np.count_nonzero(Y_with_de)}")
    print(f"Dilation-erosion effect: +{np.count_nonzero(Y_with_de) - np.count_nonzero(Y_without_de)}")
    
    print(f"\nRisk details:")
    for risk_idx, original_count in top_risks:
        risk_name = indexer.index_to_risk(risk_idx)
        original_mean = np.mean(Y_original[:, risk_idx])
        without_de_mean = np.mean(Y_without_de[:, risk_idx])
        with_de_mean = np.mean(Y_with_de[:, risk_idx])
        
        print(f"  Risk {risk_idx}: {risk_name}")
        print(f"    Original occurrences: {original_count}")
        print(f"    Original mean value: {original_mean:.4f}")
        print(f"    Without dilation-erosion mean: {without_de_mean:.4f}")
        print(f"    With dilation-erosion mean: {with_de_mean:.4f}")
        print(f"    Dilation-erosion improvement: {with_de_mean - without_de_mean:.4f}")

def main():
    """Main function to run the comparison."""
    # Configuration parameters - modify these as needed
    patient_id = "1125112810"  # Patient ID to visualize
    data_dir = "database/icu_patients"  # Directory containing patient data
    max_risks = 1  # Maximum number of risks to display per subplot
    dilation_window_hours = 48.0  # Dilation window in hours
    erosion_window_hours = 24.0    # Erosion window in hours
    risk_growth_rate = 2.0  # Risk growth rate
    risk_decay_rate = 4.0  # Risk decay rate
    
    plot_comparison(
        patient_id=patient_id,
        data_dir=data_dir,
        max_risks=max_risks,
        dilation_window_hours=dilation_window_hours,
        erosion_window_hours=erosion_window_hours,
        risk_growth_rate=risk_growth_rate,
        risk_decay_rate=risk_decay_rate,
    )

if __name__ == "__main__":
    main()
