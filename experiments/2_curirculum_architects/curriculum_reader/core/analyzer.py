"""Analysis utilities for the metadata layer."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .reader import MetadataReader, RejectionReader


@dataclass
class BandDistribution:
    """Distribution of records across curriculum bands."""

    counts: Dict[str, int]
    percentages: Dict[str, float]
    total: int


@dataclass
class MetricStats:
    """Statistics for a numeric metric column."""

    name: str
    min: float
    max: float
    mean: float
    std: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    count: int
    null_count: int


@dataclass
class DatasetSummary:
    """Summary statistics for the metadata layer."""

    total_records: int
    total_partitions: int
    columns: List[str]
    band_distribution: Optional[BandDistribution]
    modality_distribution: Optional[Dict[str, int]]
    difficulty_stats: Optional[MetricStats]


class MetadataAnalyzer:
    """Analyzer for curriculum metadata layer.

    Provides statistical analysis and visualization helpers
    for understanding the distribution of training data.
    """

    def __init__(
        self,
        metadata_reader: MetadataReader,
        rejection_reader: Optional[RejectionReader] = None,
    ):
        """Initialize analyzer.

        Args:
            metadata_reader: Reader for metadata layer
            rejection_reader: Optional reader for rejection layer
        """
        self.reader = metadata_reader
        self.rejection_reader = rejection_reader

    def get_summary(self) -> DatasetSummary:
        """Get overall dataset summary."""
        total_records = self.reader.count_rows()
        partitions = self.reader.get_partitions()
        columns = self.reader.get_column_names()

        # Try to get band distribution
        band_dist = None
        if "band_assignment_band" in columns:
            band_dist = self.get_band_distribution()

        # Try to get modality distribution
        modality_dist = None
        if "modality_primary_modality" in columns:
            modality_dist = self.get_modality_distribution()

        # Try to get difficulty stats
        difficulty_stats = None
        if "difficulty_score" in columns:
            difficulty_stats = self.get_metric_stats("difficulty_score")

        return DatasetSummary(
            total_records=total_records,
            total_partitions=len(partitions),
            columns=columns,
            band_distribution=band_dist,
            modality_distribution=modality_dist,
            difficulty_stats=difficulty_stats,
        )

    def get_band_distribution(self) -> BandDistribution:
        """Get distribution of records across curriculum bands."""
        table = self.reader.read_all(columns=["band_assignment_band"])

        # Count by band
        bands = table.column("band_assignment_band").to_pylist()
        counts: Dict[str, int] = {}
        for band in bands:
            band_str = str(band) if band is not None else "unknown"
            counts[band_str] = counts.get(band_str, 0) + 1

        total = sum(counts.values())
        percentages = {
            band: round(count / total * 100, 2) if total > 0 else 0
            for band, count in counts.items()
        }

        return BandDistribution(
            counts=counts,
            percentages=percentages,
            total=total,
        )

    def get_modality_distribution(self) -> Dict[str, int]:
        """Get distribution of records across modalities."""
        table = self.reader.read_all(columns=["modality_primary_modality"])

        modalities = table.column("modality_primary_modality").to_pylist()
        counts: Dict[str, int] = {}
        for modality in modalities:
            mod_str = str(modality) if modality is not None else "unknown"
            counts[mod_str] = counts.get(mod_str, 0) + 1

        return counts

    def get_metric_stats(self, column: str) -> MetricStats:
        """Get statistics for a numeric metric column.

        Args:
            column: Column name

        Returns:
            MetricStats with descriptive statistics
        """
        table = self.reader.read_all(columns=[column])
        col = table.column(column)

        # Handle nulls
        non_null = col.drop_null()
        null_count = len(col) - len(non_null)

        if len(non_null) == 0:
            return MetricStats(
                name=column,
                min=0.0,
                max=0.0,
                mean=0.0,
                std=0.0,
                p25=0.0,
                p50=0.0,
                p75=0.0,
                p95=0.0,
                p99=0.0,
                count=0,
                null_count=null_count,
            )

        values = non_null.to_pylist()
        values_sorted = sorted(values)
        n = len(values)

        def percentile(p: float) -> float:
            idx = int(p * n / 100)
            return values_sorted[min(idx, n - 1)]

        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_val = variance**0.5

        return MetricStats(
            name=column,
            min=min(values),
            max=max(values),
            mean=round(mean_val, 4),
            std=round(std_val, 4),
            p25=percentile(25),
            p50=percentile(50),
            p75=percentile(75),
            p95=percentile(95),
            p99=percentile(99),
            count=n,
            null_count=null_count,
        )

    def get_correlation(self, col1: str, col2: str) -> float:
        """Calculate Pearson correlation between two columns.

        Args:
            col1: First column name
            col2: Second column name

        Returns:
            Correlation coefficient
        """
        table = self.reader.read_all(columns=[col1, col2])

        v1 = table.column(col1).to_pylist()
        v2 = table.column(col2).to_pylist()

        # Filter out pairs with nulls
        pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
        if len(pairs) < 2:
            return 0.0

        x, y = zip(*pairs)
        n = len(x)

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return round(cov / (std_x * std_y), 4)

    def get_file_stats(self) -> List[Dict[str, Any]]:
        """Get per-file statistics."""
        partitions = self.reader.get_partitions()

        stats = []
        for partition in partitions:
            table = self.reader.read_partition(partition)
            row_count = len(table)

            stat = {
                "file_name": partition,
                "record_count": row_count,
            }

            # Add band distribution if available
            if "band_assignment_band" in table.schema.names:
                bands = table.column("band_assignment_band").to_pylist()
                band_counts: Dict[str, int] = {}
                for band in bands:
                    b = str(band) if band is not None else "unknown"
                    band_counts[b] = band_counts.get(b, 0) + 1
                stat["band_distribution"] = band_counts

            stats.append(stat)

        return stats

    def get_rejection_summary(self) -> Optional[Dict[str, Any]]:
        """Get rejection layer summary if available."""
        if not self.rejection_reader:
            return None

        try:
            total_rejections = self.rejection_reader.count_rows()
            by_metric = self.rejection_reader.get_rejection_counts_by_metric()
            by_reason = self.rejection_reader.get_rejection_counts_by_reason()

            return {
                "total_rejections": total_rejections,
                "by_metric": by_metric,
                "by_reason": by_reason,
            }
        except Exception as e:
            return {"error": str(e)}

    def compare_bands(self, metric: str) -> Dict[str, MetricStats]:
        """Compare metric statistics across bands.

        Args:
            metric: Metric column to analyze

        Returns:
            Dictionary mapping band to MetricStats
        """
        if "band_assignment_band" not in self.reader.get_column_names():
            return {}

        table = self.reader.read_all(columns=["band_assignment_band", metric])

        # Group by band
        bands_data: Dict[str, List[float]] = {}
        for row in table.to_pylist():
            band = str(row.get("band_assignment_band", "unknown"))
            value = row.get(metric)

            if value is not None:
                if band not in bands_data:
                    bands_data[band] = []
                bands_data[band].append(value)

        # Calculate stats per band
        results = {}
        for band, values in bands_data.items():
            if not values:
                continue

            values_sorted = sorted(values)
            n = len(values)

            def percentile(p: float) -> float:
                idx = int(p * n / 100)
                return values_sorted[min(idx, n - 1)]

            mean_val = sum(values) / n
            variance = sum((x - mean_val) ** 2 for x in values) / n

            results[band] = MetricStats(
                name=f"{metric}@{band}",
                min=min(values),
                max=max(values),
                mean=round(mean_val, 4),
                std=round(variance**0.5, 4),
                p25=percentile(25),
                p50=percentile(50),
                p75=percentile(75),
                p95=percentile(95),
                p99=percentile(99),
                count=n,
                null_count=0,
            )

        return results

    def export_summary_report(self) -> str:
        """Generate a text summary report."""
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "CURRICULUM METADATA ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Total Records: {summary.total_records:,}",
            f"Total Partitions: {summary.total_partitions}",
            f"Columns: {len(summary.columns)}",
            "",
        ]

        if summary.band_distribution:
            lines.extend(
                [
                    "BAND DISTRIBUTION",
                    "-" * 40,
                ]
            )
            for band in sorted(summary.band_distribution.counts.keys()):
                count = summary.band_distribution.counts[band]
                pct = summary.band_distribution.percentages[band]
                lines.append(f"  {band}: {count:,} ({pct}%)")
            lines.append("")

        if summary.modality_distribution:
            lines.extend(
                [
                    "MODALITY DISTRIBUTION",
                    "-" * 40,
                ]
            )
            for mod, count in sorted(
                summary.modality_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {mod}: {count:,}")
            lines.append("")

        if summary.difficulty_stats:
            stats = summary.difficulty_stats
            lines.extend(
                [
                    "DIFFICULTY SCORE STATISTICS",
                    "-" * 40,
                    f"  Mean: {stats.mean}",
                    f"  Std:  {stats.std}",
                    f"  Min:  {stats.min}",
                    f"  Max:  {stats.max}",
                    f"  P50:  {stats.p50}",
                    f"  P95:  {stats.p95}",
                    "",
                ]
            )

        rejection_summary = self.get_rejection_summary()
        if rejection_summary and "total_rejections" in rejection_summary:
            lines.extend(
                [
                    "REJECTION SUMMARY",
                    "-" * 40,
                    f"  Total Rejections: {rejection_summary['total_rejections']:,}",
                ]
            )
            if rejection_summary.get("by_metric"):
                lines.append("  By Metric:")
                for metric, count in rejection_summary["by_metric"].items():
                    lines.append(f"    {metric}: {count:,}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
