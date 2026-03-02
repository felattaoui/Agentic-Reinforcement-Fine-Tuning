"""
Matplotlib chart builders for notebook display.
"""

from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# Standard color palette for models
MODEL_COLORS = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#2ecc71']


def plot_metrics_comparison(
    all_results: Dict[str, pd.DataFrame],
    col_mapping: Dict[str, str],
    output_path: Path,
    figsize: tuple = (14, 6),
    fontsize: int = 8,
    decimals: int = 3
) -> None:
    """
    Plot bar chart comparing Recall/Precision/F2 across models.

    Args:
        all_results: Dict mapping model name to DataFrame with evaluation results
        col_mapping: Dict mapping metric name to column name (e.g., {'recall': 'outputs.planner.recall'})
        output_path: Path to save the figure
        figsize: Figure size tuple
        fontsize: Font size for value labels
        decimals: Number of decimal places for value labels
    """
    fig, ax = plt.subplots(figsize=figsize)

    metrics = ['Recall', 'Precision', 'F2']
    model_names = list(all_results.keys())
    colors = MODEL_COLORS[:len(model_names)]

    # Get scores for each model
    scores_by_model = []
    for name, df in all_results.items():
        scores_by_model.append([df[col_mapping[m.lower()]].mean() for m in metrics])

    x = range(len(metrics))
    width = 0.12

    for i, (name, scores, color) in enumerate(zip(model_names, scores_by_model, colors)):
        offset = (i - 2.5) * width
        bars = ax.bar([xi + offset for xi in x], scores, width, label=name, color=color)

        # Add values on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{score:.{decimals}f}',
                ha='center',
                va='bottom',
                fontsize=fontsize,
                rotation=45
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.5, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('RFT Planner Evaluation: o4-mini (vanilla) vs gpt-5.2 (all reasoning levels) vs o4-mini (fine-tuned)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\n📊 Saved: {output_path}")


def plot_react_distributions(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (15, 4)
) -> None:
    """
    Plot 3 histograms: F2 distribution, iterations per case, tools passed.

    Args:
        df: DataFrame with columns: f2, iterations, tools_passed
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # F2 histogram
    axes[0].hist(df['f2'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['f2'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['f2'].mean():.3f}")
    axes[0].set_xlabel('F2 Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('F2 Score Distribution')
    axes[0].legend()

    # Iterations histogram
    axes[1].hist(df['iterations'], bins=range(0, int(df['iterations'].max()) + 2),
                 edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(df['iterations'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['iterations'].mean():.1f}")
    axes[1].set_xlabel('Iterations (tool calls)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('ReAct Iterations per Case')
    axes[1].legend()

    # Tools passed histogram
    axes[2].hist(df['tools_passed'], bins=range(0, 16),
                 edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(df['tools_passed'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['tools_passed'].mean():.1f}")
    axes[2].set_xlabel('Tools Passed to Executor')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Dynamic Tool Filtering')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\n📊 Saved: {output_path}")

def plot_cost_vs_performance(
    f2_scores: Dict[str, float],
    cost_results: Dict[str, Dict[str, float]],
    output_path: Path,
    figsize: tuple = (14, 5)
) -> None:
    """
    Plot cost analysis: bar chart + scatter plot (F2 vs Cost).

    Args:
        f2_scores: Dict mapping model name to F2 score
        cost_results: Dict mapping model name to cost dict with 'inference_cost_per_1k'
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    model_names = [
        "o4-mini (vanilla)",
        "gpt-5.2 (none)",
        "gpt-5.2 (low)",
        "gpt-5.2 (medium)",
        "gpt-5.2 (high)",
        "o4-mini (fine-tuned)"
    ]
    # Filter to only models that exist
    model_names = [m for m in model_names if m in cost_results]
    colors = MODEL_COLORS[:len(model_names)]

    # Get costs
    costs = [cost_results[name]["inference_cost_per_1k"] for name in model_names]

    # Chart 1: Cost comparison (bar chart)
    ax1 = axes[0]
    bars = ax1.bar(model_names, costs, color=colors)
    ax1.set_ylabel('Cost per 1,000 requests (USD)')
    ax1.set_title('Inference Cost Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'${cost:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Chart 2: F2 vs Cost scatter plot
    ax2 = axes[1]
    for i, name in enumerate(model_names):
        f2 = f2_scores[name]
        cost = cost_results[name]["inference_cost_per_1k"]
        ax2.scatter(cost, f2, s=150, c=colors[i], label=name, edgecolors='black', linewidth=1)
        ax2.annotate(
            name,
            (cost, f2),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            ha='left'
        )

    ax2.set_xlabel('Cost per 1,000 requests (USD)')
    ax2.set_ylabel('F2 Score')
    ax2.set_title('Performance vs Cost Trade-off')
    ax2.grid(True, alpha=0.3)

    # Highlight Fine-tuned position
    if "o4-mini (fine-tuned)" in f2_scores and "o4-mini (fine-tuned)" in cost_results:
        ax2.axhline(y=f2_scores["o4-mini (fine-tuned)"], color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=cost_results["o4-mini (fine-tuned)"]["inference_cost_per_1k"], color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\n📊 Saved: {output_path}")


def plot_latency_analysis(
    f2_scores: Dict[str, float],
    usage_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    figsize: tuple = (14, 5)
) -> None:
    """
    Plot latency analysis: bar chart + scatter plot (F2 vs Latency).

    Args:
        f2_scores: Dict mapping model name to F2 score
        usage_data: Dict mapping model name to usage dict with 'avg_latency_ms'
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    model_names = [
        "o4-mini (vanilla)",
        "gpt-5.2 (none)",
        "gpt-5.2 (low)",
        "gpt-5.2 (medium)",
        "gpt-5.2 (high)",
        "o4-mini (fine-tuned)"
    ]
    model_names = [m for m in model_names if m in usage_data]
    colors = MODEL_COLORS[:len(model_names)]

    latencies = [usage_data[name].get("avg_latency_ms", 0) for name in model_names]

    # Chart 1: Latency bar chart
    ax1 = axes[0]
    bars = ax1.bar(model_names, latencies, color=colors)
    ax1.set_ylabel('Average Latency (ms)')
    ax1.set_title('Response Latency Comparison')
    ax1.tick_params(axis='x', rotation=45)

    for bar, lat in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f'{lat:,.0f}ms',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Chart 2: F2 vs Latency scatter
    ax2 = axes[1]
    for i, name in enumerate(model_names):
        f2 = f2_scores.get(name, 0)
        lat = usage_data[name].get("avg_latency_ms", 0)
        ax2.scatter(lat, f2, s=150, c=colors[i], label=name, edgecolors='black', linewidth=1)
        ax2.annotate(
            name, (lat, f2),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, ha='left'
        )

    ax2.set_xlabel('Average Latency (ms)')
    ax2.set_ylabel('F2 Score')
    ax2.set_title('Performance vs Latency Trade-off')
    ax2.grid(True, alpha=0.3)

    if "o4-mini (fine-tuned)" in f2_scores and "o4-mini (fine-tuned)" in usage_data:
        ax2.axhline(y=f2_scores["o4-mini (fine-tuned)"], color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=usage_data["o4-mini (fine-tuned)"].get("avg_latency_ms", 0),
                    color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\n📊 Saved: {output_path}")
