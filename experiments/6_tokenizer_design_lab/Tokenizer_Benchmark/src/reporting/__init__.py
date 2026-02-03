"""Reporting package for benchmark results."""

from .tables import (
    generate_comparison_table,
    generate_metrics_table,
    format_as_markdown,
    format_as_html,
    generate_report,
)
from .charts import (
    create_compression_chart,
    create_fertility_chart,
    create_radar_chart,
    save_all_charts,
)

__all__ = [
    'generate_comparison_table',
    'generate_metrics_table',
    'format_as_markdown',
    'format_as_html',
    'create_compression_chart',
    'create_fertility_chart',
    'create_radar_chart',
    'save_all_charts',
]
