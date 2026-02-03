"""
Table Generation - Create comparison tables for tokenizer benchmarks.

Supports Markdown and HTML output formats.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_comparison_table(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Generate a comparison table across tokenizers.
    
    Args:
        results: Dict of tokenizer_name -> metrics_dict
        metrics: List of metric names to include
    
    Returns:
        Table data structure
    """
    if not results:
        return {'headers': [], 'rows': []}
    
    # Default metrics if not specified
    if metrics is None:
        metrics = [
            'tokens_per_byte',
            'tokens_per_char',
            'compression_ratio',
            'fertility',
        ]
    
    # Build table
    headers = ['Tokenizer'] + metrics
    rows = []
    
    for tok_name, tok_results in results.items():
        row = [tok_name]
        for metric in metrics:
            value = tok_results.get(metric, {})
            if isinstance(value, dict) and 'mean' in value:
                row.append(f"{value['mean']:.4f}")
            elif isinstance(value, (int, float)):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value) if value else 'N/A')
        rows.append(row)
    
    return {'headers': headers, 'rows': rows}


def generate_metrics_table(
    metrics: Dict[str, Any],
    title: str = "Metrics"
) -> Dict[str, Any]:
    """
    Generate a table for a single tokenizer's metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
        title: Table title
    
    Returns:
        Table data structure
    """
    headers = ['Metric', 'Value', 'Min', 'Max']
    rows = []
    
    for name, value in metrics.items():
        if isinstance(value, dict):
            if 'mean' in value:
                rows.append([
                    name,
                    f"{value.get('mean', 0):.4f}",
                    f"{value.get('min', 0):.4f}",
                    f"{value.get('max', 0):.4f}",
                ])
            else:
                rows.append([name, str(value), '', ''])
        elif isinstance(value, (int, float)):
            rows.append([name, f"{value:.4f}", '', ''])
        else:
            rows.append([name, str(value), '', ''])
    
    return {'title': title, 'headers': headers, 'rows': rows}


def format_as_markdown(table: Dict[str, Any]) -> str:
    """
    Format table as Markdown.
    
    Args:
        table: Table data from generate_*_table functions
    
    Returns:
        Markdown string
    """
    lines = []
    
    if 'title' in table:
        lines.append(f"### {table['title']}")
        lines.append("")
    
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    if not headers:
        return ""
    
    # Header row
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    
    # Data rows
    for row in rows:
        lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    
    return '\n'.join(lines)


def format_as_html(table: Dict[str, Any], style: str = "modern") -> str:
    """
    Format table as HTML.
    
    Args:
        table: Table data from generate_*_table functions
        style: CSS style preset ('modern', 'minimal', 'classic')
    
    Returns:
        HTML string
    """
    styles = {
        'modern': """
<style>
.benchmark-table {
    border-collapse: collapse;
    width: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 20px 0;
}
.benchmark-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 15px;
    text-align: left;
    font-weight: 600;
}
.benchmark-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #e0e0e0;
}
.benchmark-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.benchmark-table tr:hover {
    background-color: #e8f4f8;
}
</style>
""",
        'minimal': """
<style>
.benchmark-table {
    border-collapse: collapse;
    font-family: monospace;
}
.benchmark-table th, .benchmark-table td {
    padding: 8px 12px;
    border: 1px solid #ddd;
}
.benchmark-table th {
    background: #f0f0f0;
}
</style>
""",
        'classic': """
<style>
.benchmark-table {
    border-collapse: collapse;
    border: 2px solid #333;
}
.benchmark-table th {
    background: #333;
    color: white;
    padding: 10px;
}
.benchmark-table td {
    padding: 8px;
    border: 1px solid #333;
}
</style>
""",
    }
    
    css = styles.get(style, styles['modern'])
    
    html_parts = [css, '<table class="benchmark-table">']
    
    if 'title' in table:
        html_parts.insert(0, f"<h3>{table['title']}</h3>")
    
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    # Header row
    html_parts.append('<thead><tr>')
    for header in headers:
        html_parts.append(f'<th>{header}</th>')
    html_parts.append('</tr></thead>')
    
    # Data rows
    html_parts.append('<tbody>')
    for row in rows:
        html_parts.append('<tr>')
        for cell in row:
            html_parts.append(f'<td>{cell}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    
    html_parts.append('</table>')
    
    return '\n'.join(html_parts)


def generate_report(
    results: Dict[str, Dict[str, Any]],
    validation_results: Dict[str, Any] = None,
    output_format: str = "markdown"
) -> str:
    """
    Generate a complete benchmark report.
    
    Args:
        results: Benchmark results per tokenizer
        validation_results: Optional validation results
        output_format: 'markdown' or 'html'
    
    Returns:
        Complete report string
    """
    formatter = format_as_markdown if output_format == "markdown" else format_as_html
    sections = []
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if output_format == "markdown":
        sections.append(f"# Tokenizer Benchmark Report")
        sections.append(f"*Generated: {timestamp}*")
        sections.append("")
    else:
        sections.append(f"<h1>Tokenizer Benchmark Report</h1>")
        sections.append(f"<p><em>Generated: {timestamp}</em></p>")
    
    # Compression metrics
    comp_table = generate_comparison_table(
        {k: v.get('compression', {}) for k, v in results.items()},
        metrics=['tokens_per_byte', 'compression_ratio']
    )
    if output_format == "markdown":
        sections.append("## Compression Metrics")
    else:
        sections.append("<h2>Compression Metrics</h2>")
    sections.append(formatter(comp_table))
    sections.append("")
    
    # Fertility metrics (if present)
    if any('fertility' in v for v in results.values()):
        fert_table = generate_comparison_table(
            {k: v.get('fertility', {}) for k, v in results.items()},
            metrics=['fertility', 'num_words', 'num_tokens']
        )
        if output_format == "markdown":
            sections.append("## Fertility Metrics")
        else:
            sections.append("<h2>Fertility Metrics</h2>")
        sections.append(formatter(fert_table))
        sections.append("")
    
    # Speed metrics (if present)
    if any('speed' in v for v in results.values()):
        speed_data = {}
        for k, v in results.items():
            if 'speed' in v:
                speed_data[k] = {
                    'encode_tokens_per_sec': v['speed'].get('encoding', {}).get('tokens_per_second', 0),
                    'decode_tokens_per_sec': v['speed'].get('decoding', {}).get('tokens_per_second', 0),
                }
        
        speed_table = generate_comparison_table(
            speed_data,
            metrics=['encode_tokens_per_sec', 'decode_tokens_per_sec']
        )
        if output_format == "markdown":
            sections.append("## Speed Metrics")
        else:
            sections.append("<h2>Speed Metrics</h2>")
        sections.append(formatter(speed_table))
        sections.append("")
    
    # Validation results
    if validation_results:
        if output_format == "markdown":
            sections.append("## Validation Results")
            sections.append("")
            for check_name, check_result in validation_results.items():
                status = "✅ PASS" if check_result.get('passed', False) else "❌ FAIL"
                sections.append(f"- **{check_name}**: {status}")
                if check_result.get('issues'):
                    for issue in check_result['issues'][:3]:
                        sections.append(f"  - {issue}")
        else:
            sections.append("<h2>Validation Results</h2><ul>")
            for check_name, check_result in validation_results.items():
                status = "✅ PASS" if check_result.get('passed', False) else "❌ FAIL"
                sections.append(f"<li><strong>{check_name}</strong>: {status}</li>")
            sections.append("</ul>")
    
    return '\n'.join(sections)
