"""
Chart Generation - Create visualizations for tokenizer benchmarks.

Uses matplotlib for generating comparison charts.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Lazy import matplotlib to avoid issues if not installed
def _get_plt():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        return plt
    except ImportError:
        return None


def create_compression_chart(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Compression Ratio Comparison"
) -> Optional[str]:
    """
    Create a bar chart comparing compression ratios.
    
    Args:
        results: Dict of tokenizer_name -> metrics
        output_path: Path to save chart (optional)
        title: Chart title
    
    Returns:
        Path to saved chart, or None if matplotlib unavailable
    """
    plt = _get_plt()
    if plt is None:
        print("Warning: matplotlib not available, skipping chart generation")
        return None
    
    tokenizers = list(results.keys())
    compression_ratios = []
    
    for tok in tokenizers:
        metrics = results[tok].get('compression', results[tok])
        if isinstance(metrics, dict):
            cr = metrics.get('compression_ratio', {})
            if isinstance(cr, dict):
                compression_ratios.append(cr.get('mean', 0))
            else:
                compression_ratios.append(cr)
        else:
            compression_ratios.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with gradient colors
    colors = plt.cm.viridis([i / len(tokenizers) for i in range(len(tokenizers))])
    bars = ax.bar(tokenizers, compression_ratios, color=colors, edgecolor='white', linewidth=1.2)
    
    # Styling
    ax.set_ylabel('Bytes per Token (higher is better)', fontsize=12)
    ax.set_xlabel('Tokenizer', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, val in zip(bars, compression_ratios):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    plt.close()
    return None


def create_fertility_chart(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Fertility Comparison (Tokens per Word)"
) -> Optional[str]:
    """
    Create a bar chart comparing fertility across tokenizers.
    
    Args:
        results: Dict of tokenizer_name -> metrics
        output_path: Path to save chart
        title: Chart title
    
    Returns:
        Path to saved chart
    """
    plt = _get_plt()
    if plt is None:
        return None
    
    tokenizers = list(results.keys())
    fertilities = []
    
    for tok in tokenizers:
        metrics = results[tok].get('fertility', results[tok])
        if isinstance(metrics, dict):
            fert = metrics.get('fertility', 0)
        else:
            fert = metrics if isinstance(metrics, (int, float)) else 0
        fertilities.append(fert)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use different colormap
    colors = plt.cm.plasma([i / len(tokenizers) for i in range(len(tokenizers))])
    bars = ax.bar(tokenizers, fertilities, color=colors, edgecolor='white', linewidth=1.2)
    
    ax.set_ylabel('Tokens per Word (lower is better)', fontsize=12)
    ax.set_xlabel('Tokenizer', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, fertilities):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    plt.close()
    return None


def create_radar_chart(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Multi-Metric Comparison"
) -> Optional[str]:
    """
    Create a radar/spider chart for multi-metric comparison.
    
    Args:
        results: Dict of tokenizer_name -> {metric_name: normalized_score}
                 Scores should be normalized to 0-1 range
        output_path: Path to save chart
        title: Chart title
    
    Returns:
        Path to saved chart
    """
    plt = _get_plt()
    if plt is None:
        return None
    
    import numpy as np
    
    if not results:
        return None
    
    # Get all metrics from first tokenizer
    first_tok = next(iter(results.values()))
    metrics = list(first_tok.keys())
    num_metrics = len(metrics)
    
    if num_metrics < 3:
        return None  # Need at least 3 metrics for radar chart
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for idx, (tok_name, tok_metrics) in enumerate(results.items()):
        values = [tok_metrics.get(m, 0) for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=tok_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    plt.close()
    return None


def create_language_comparison_chart(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Fertility by Language"
) -> Optional[str]:
    """
    Create grouped bar chart comparing fertility across languages.
    
    Args:
        results: Dict of tokenizer_name -> {language: fertility}
        output_path: Path to save chart
        title: Chart title
    
    Returns:
        Path to saved chart
    """
    plt = _get_plt()
    if plt is None:
        return None
    
    import numpy as np
    
    tokenizers = list(results.keys())
    if not tokenizers:
        return None
    
    languages = list(results[tokenizers[0]].keys())
    
    x = np.arange(len(languages))
    width = 0.8 / len(tokenizers)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(tokenizers)))
    
    for i, tok_name in enumerate(tokenizers):
        values = [results[tok_name].get(lang, 0) for lang in languages]
        offset = width * (i - len(tokenizers) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=tok_name, color=colors[i])
    
    ax.set_ylabel('Fertility (Tokens per Word)', fontsize=12)
    ax.set_xlabel('Language', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    plt.close()
    return None


def save_all_charts(
    results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> List[str]:
    """
    Generate and save all charts to a directory.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save charts
    
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    # Compression chart
    comp_path = str(output_path / "compression_comparison.png")
    if create_compression_chart(results, comp_path):
        saved.append(comp_path)
    
    # Fertility chart
    fert_path = str(output_path / "fertility_comparison.png")
    if create_fertility_chart(results, fert_path):
        saved.append(fert_path)
    
    return saved
