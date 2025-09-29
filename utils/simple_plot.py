from typing import Dict, Iterable, Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np


def save_line_plot_svg(
    history: Iterable[Tuple[int, float]],
    path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    success_iteration: Optional[int] = None,
) -> None:
    """Saves a line plot of loss vs. iteration."""
    iterations, losses = zip(*history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, marker='o', linestyle='-')
    
    if success_iteration and success_iteration > 0:
        success_loss = np.interp(success_iteration, iterations, losses)
        plt.axvline(x=success_iteration, color='r', linestyle='--', label=f'Success at iter {success_iteration}')
        plt.text(success_iteration + 1, success_loss, f'Success!', color='r')
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()
    print(f"Saved plot to {path}")


def save_asr_plots_svg(
    final_asr_data: Dict[str, float],
    asr_over_time_data: Optional[Dict[str, List[Tuple[int, float]]]],
    path: str,
    title: str,
) -> None:
    """Saves a combined plot of final ASR (bar) and ASR vs. Iterations (line)."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # --- Subplot 1: Final ASR Bar Chart ---
    labels = list(final_asr_data.keys())
    values = list(final_asr_data.values())
    bars = ax1.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title("Final Attack Success Rate (ASR)")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1%}', va='bottom', ha='center')

    # --- Subplot 2: ASR vs. Iterations Line Chart ---
    ax2.set_title("ASR vs. Iterations")
    if asr_over_time_data:
        has_data = False
        for model_name, data_points in asr_over_time_data.items():
            if data_points:
                has_data = True
                iterations, asr_values = zip(*sorted(data_points))
                ax2.plot(iterations, asr_values, marker='o', linestyle='-', label=model_name)
        if has_data:
            ax2.legend()

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, format="svg")
    plt.close()
    print(f"Saved ASR plot to {path}")