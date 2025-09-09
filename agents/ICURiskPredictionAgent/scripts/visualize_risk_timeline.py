#!/usr/bin/env python3
"""
Visualize risk timeline for ICU patient data.

This script loads patient data using ICUSequenceDataLoader and creates
visualizations showing how different risks evolve over time.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Any

# Add parent directory to path to import ICUSequenceDataLoader
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ICURiskPredictionAgent.icu_sequence_dataloader import ICUSequenceDataLoader
from agents.ICURiskPredictionAgent.risk_label_indexer import RiskLabelIndexer


def get_distinct_colors(n: int) -> List[str]:
    """Generate n distinct colors for plotting."""
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 1, n))
    elif n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, n))
    else:
        # For more than 20 risks, use a colormap with more colors
        return plt.cm.viridis(np.linspace(0, 1, n))


def plot_risk_timeline(
    patient_id: str,
    data_dir: str = "database/icu_patients",
    max_risks: int = 20,
    figsize: tuple = (15, 10)
):
    """
    Plot risk timeline for a specific patient.
    
    Args:
        patient_id: Patient ID to visualize
        data_dir: Directory containing patient data
        max_risks: Maximum number of risks to display (to avoid overcrowding)
        figsize: Figure size for the plot
    """
    # Load data
    print(f"Loading data for patient {patient_id}...")
    loader = ICUSequenceDataLoader(
        data_dir=data_dir,
        only_patient_id=patient_id,
        train_ratio=0.6,
        shuffle_patients=False,
    )
    
    # Get risk indexer for risk names
    indexer = RiskLabelIndexer()
    
    # Load patient data
    sample = None
    for sample in loader.iter_patient_sequences("train"):
        break
    
    if sample is None:
        print(f"No data found for patient {patient_id}")
        return
    
    # Extract data
    Y_original = sample["labels"]  # Original sharp labels
    Y_smoothed = sample["labels_smoothed"]  # Smoothed labels
    timestamps = sample["timestamps"]
    time_deltas = sample["time_deltas"]
    
    print(f"Data loaded: {Y_original.shape[0]} events, {Y_original.shape[1]} risks")
    
    # Find risks that have at least one occurrence
    risk_occurrences = np.any(Y_original > 0, axis=0)
    active_risk_indices = np.where(risk_occurrences)[0]
    
    print(f"Found {len(active_risk_indices)} risks with occurrences")
    
    if len(active_risk_indices) == 0:
        print("No risks found in the data")
        return
    
    # Limit number of risks to display
    if len(active_risk_indices) > max_risks:
        # Sort by total occurrence count and take top risks
        risk_counts = np.sum(Y_original[:, active_risk_indices] > 0, axis=0)
        top_risk_indices = np.argsort(risk_counts)[-max_risks:]
        active_risk_indices = active_risk_indices[top_risk_indices]
        print(f"Displaying top {max_risks} most frequent risks")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Get colors for each risk
    colors = get_distinct_colors(len(active_risk_indices))
    
    # Plot original labels
    ax1.set_title(f"Original Risk Labels - Patient {patient_id}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Risk Value", fontsize=12)
    ax1.set_xlabel("Event Index", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for i, risk_idx in enumerate(active_risk_indices):
        risk_values = Y_original[:, risk_idx]
        # Only plot non-zero segments to make it cleaner
        non_zero_mask = risk_values > 0
        if np.any(non_zero_mask):
            event_indices = np.arange(len(risk_values))[non_zero_mask]
            values = risk_values[non_zero_mask]
            ax1.plot(event_indices, values, 'o-', color=colors[i], 
                    label=f"Risk {risk_idx}", markersize=2, linewidth=1)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot smoothed labels
    ax2.set_title(f"Smoothed Risk Labels - Patient {patient_id}", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Risk Value", fontsize=12)
    ax2.set_xlabel("Event Index", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, risk_idx in enumerate(active_risk_indices):
        risk_values = Y_smoothed[:, risk_idx]
        # Only plot if there are non-zero values
        if np.any(risk_values > 0):
            ax2.plot(risk_values, '-', color=colors[i], 
                    label=f"Risk {risk_idx}", linewidth=1.5)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_ylim(-0.1, 1.1)
    
    # Add risk names to legend (if not too many)
    if len(active_risk_indices) <= 10:
        risk_names = [indexer.index_to_risk(idx) for idx in active_risk_indices]
        # Create a separate legend with risk names
        fig.legend(risk_names, bbox_to_anchor=(0.02, 0.5), loc='center left', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_path = f"risk_timeline_patient_{patient_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary for Patient {patient_id}:")
    print(f"Total events: {len(timestamps)}")
    print(f"Active risks: {len(active_risk_indices)}")
    print(f"Original non-zero values: {np.count_nonzero(Y_original)}")
    print(f"Smoothed non-zero values: {np.count_nonzero(Y_smoothed)}")
    
    # Show risk details
    print(f"\nRisk details:")
    for i, risk_idx in enumerate(active_risk_indices):
        risk_name = indexer.index_to_risk(risk_idx)
        original_count = np.count_nonzero(Y_original[:, risk_idx])
        smoothed_mean = np.mean(Y_smoothed[:, risk_idx])
        print(f"  Risk {risk_idx}: {risk_name}")
        print(f"    Original occurrences: {original_count}")
        print(f"    Smoothed mean value: {smoothed_mean:.4f}")


def plot_individual_risk(patient_id: str, risk_index: int, data_dir: str = "database/icu_patients"):
    """Plot a specific risk in detail."""
    print(f"Loading data for patient {patient_id}, risk {risk_index}...")
    
    loader = ICUSequenceDataLoader(
        data_dir=data_dir,
        only_patient_id=patient_id,
        train_ratio=0.6,
        shuffle_patients=False,
    )
    
    indexer = RiskLabelIndexer()
    
    for sample in loader.iter_patient_sequences("train"):
        Y_original = sample["labels"]
        Y_smoothed = sample["labels_smoothed"]
        timestamps = sample["timestamps"]
        
        if risk_index >= Y_original.shape[1]:
            print(f"Risk index {risk_index} out of range (max: {Y_original.shape[1]-1})")
            return
        
        risk_name = indexer.index_to_risk(risk_index)
        print(f"Risk name: {risk_name}")
        
        # Check if risk has any occurrences
        if not np.any(Y_original[:, risk_index] > 0):
            print(f"Risk {risk_index} has no occurrences")
            return
        
        # Create detailed plot
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot both original and smoothed
        ax.plot(Y_original[:, risk_index], 'o-', label='Original', markersize=3, alpha=0.7)
        ax.plot(Y_smoothed[:, risk_index], '-', label='Smoothed', linewidth=2)
        
        ax.set_title(f"Risk {risk_index}: {risk_name} - Patient {patient_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Event Index", fontsize=12)
        ax.set_ylabel("Risk Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        # Save plot
        output_path = f"risk_{risk_index}_patient_{patient_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Individual risk plot saved as: {output_path}")
        
        plt.show()
        break


if __name__ == "__main__":
    # Default patient ID
    patient_id = "1125112810"
    
    # Check if patient ID provided as command line argument
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    
    print(f"Visualizing risk timeline for patient: {patient_id}")
    
    try:
        # Main visualization
        plot_risk_timeline(patient_id, max_risks=1)
        
        # Optional: Plot individual risks
        # Uncomment the following lines to plot specific risks in detail
        # plot_individual_risk(patient_id, 0)  # First risk
        # plot_individual_risk(patient_id, 1)  # Second risk
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
