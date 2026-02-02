"""Dataset reader for downloaded files (JSON and Parquet)."""

import json
from pathlib import Path
from typing import Iterator, Literal, Optional

import pandas as pd
import pyarrow.parquet as pq

# Base data directory
DATA_DIR = Path(".data")

# Supported output formats
OutputFormat = Literal["json", "parquet"]

# Dataset display names
DATASET_NAMES = {
    "dolma": "Dolma v1.7",
    "sangraha": "Sangraha",
    "indiccorp": "IndicCorp v2",
    "ncert": "NCERT",
}


class DatasetReader:
    """Reader for downloaded dataset files (JSON and Parquet)."""
    
    def __init__(
        self,
        dataset: str,
        subfolder: str,
        output_format: OutputFormat = "parquet",
    ):
        """Initialize dataset reader.
        
        Args:
            dataset: Dataset name ('dolma', 'sangraha', 'indiccorp', 'ncert')
            subfolder: Subfolder name (source for dolma, language for sangraha/indiccorp, '_all' for ncert)
            output_format: Output format to read ('json' or 'parquet')
        """
        self.dataset = dataset
        self.subfolder = subfolder
        self.output_format = output_format
        self.base_path = DATA_DIR / dataset / subfolder / output_format
    
    @property
    def name(self) -> str:
        """Get the display name of the dataset.
        
        Returns:
            Human-readable dataset name
        """
        return DATASET_NAMES.get(self.dataset, self.dataset)
    
    def get_files(self) -> list[Path]:
        """Get list of all data files.
        
        Returns:
            List of Path objects for all data files
        """
        if not self.base_path.exists():
            return []
        
        ext = "parquet" if self.output_format == "parquet" else "json"
        files = sorted(self.base_path.glob(f"*.{ext}"))
        return [f for f in files if not f.name.startswith(".")]
    
    def read_file(self, file_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
        """Read a single data file.
        
        Args:
            file_path: Path to the file
            limit: Maximum number of records to read (None for all)
            
        Returns:
            DataFrame with records
        """
        if self.output_format == "parquet":
            if limit is not None:
                # Read only required rows for efficiency
                table = pq.read_table(file_path)
                df = table.slice(0, min(limit, table.num_rows)).to_pandas()
            else:
                df = pd.read_parquet(file_path)
        else:
            df = pd.read_json(file_path)
            if limit is not None:
                df = df.head(limit)
        
        # Parse metadata JSON string back to dict for parquet files
        if self.output_format == "parquet" and "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return df
    
    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Read data files with optional limit.
        
        Args:
            limit: Maximum number of records to read (None for all)
            
        Returns:
            DataFrame with records
        """
        files = self.get_files()
        if not files:
            return pd.DataFrame()
        
        if limit is None:
            # Read all
            dfs = [self.read_file(f) for f in files]
            return pd.concat(dfs, ignore_index=True)
        
        # Read with limit
        dfs = []
        remaining = limit
        for f in files:
            if remaining <= 0:
                break
            df = self.read_file(f, limit=remaining)
            dfs.append(df)
            remaining -= len(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    
    def read_all(self) -> pd.DataFrame:
        """Read all data files and concatenate.
        
        Returns:
            DataFrame with all records
        """
        return self.read(limit=None)
    
    def iter_records(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Iterate over records one at a time (memory-efficient).
        
        Args:
            limit: Maximum number of records to yield (None for all)
        
        Yields:
            dict: Record dictionary
        """
        count = 0
        for file_path in self.get_files():
            df = self.read_file(file_path)
            for _, row in df.iterrows():
                if limit is not None and count >= limit:
                    return
                yield row.to_dict()
                count += 1
    
    def get_schema(self) -> dict:
        """Get schema information from the first file.
        
        Returns:
            dict with schema (column types) and rows count
        """
        files = self.get_files()
        if not files:
            return {}
        
        if self.output_format == "parquet":
            table = pq.read_table(files[0])
            return {
                "schema": {name: str(table.schema.field(name).type) for name in table.column_names},
                "rows": table.num_rows,
            }
        else:
            df = pd.read_json(files[0])
            return {
                "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "rows": len(df),
            }
    
    def get_stats(self) -> dict:
        """Get statistics about the dataset.
        
        Returns:
            dict with file count, total records, and size
        """
        files = self.get_files()
        if not files:
            return {"files": 0, "records": 0, "size_bytes": 0, "size_mb": 0}
        
        total_size = sum(f.stat().st_size for f in files)
        
        # Count records
        total_records = 0
        for f in files:
            if self.output_format == "parquet":
                table = pq.read_table(f)
                total_records += table.num_rows
            else:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    total_records += len(data)
        
        return {
            "files": len(files),
            "records": total_records,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
        }
    
    def sample(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of records.
        
        Args:
            n: Number of records to sample
            
        Returns:
            DataFrame with sample records
        """
        files = self.get_files()
        if not files:
            return pd.DataFrame()
        
        df = self.read_file(files[0])
        return df.head(n)
    
    def to_json(
        self,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        indent: int = 2,
    ) -> Optional[str]:
        """Export data to JSON file or return as JSON string.
        
        Args:
            output_path: Path to save JSON file (if None, returns JSON string)
            limit: Maximum number of records to export (None for all)
            indent: JSON indentation (default: 2)
            
        Returns:
            JSON string if output_path is None, else None (writes to file)
        """
        df = self.read(limit=limit)
        if df.empty:
            records = []
        else:
            records = df.to_dict(orient="records")
        
        json_str = json.dumps(records, ensure_ascii=False, indent=indent)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Exported {len(records)} records to {output_path}")
            return None
        
        return json_str
    
    def to_jsonl(
        self,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[str]:
        """Export data to JSON Lines format (one JSON object per line).
        
        Args:
            output_path: Path to save JSONL file (if None, returns JSONL string)
            limit: Maximum number of records to export (None for all)
            
        Returns:
            JSONL string if output_path is None, else None (writes to file)
        """
        lines = []
        for record in self.iter_records(limit=limit):
            lines.append(json.dumps(record, ensure_ascii=False))
        
        jsonl_str = "\n".join(lines)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(jsonl_str)
            print(f"Exported {len(lines)} records to {output_path}")
            return None
        
        return jsonl_str
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DatasetReader(dataset='{self.dataset}', subfolder='{self.subfolder}', "
            f"format='{self.output_format}', files={stats['files']}, records={stats['records']})"
        )


def main():
    """CLI for reading parquet files and optionally converting to JSON."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Read parquet dataset files and convert to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python parquet_reader.py --dataset dolma --subfolder gutenberg
  uv run python parquet_reader.py --dataset sangraha --subfolder hin --format json --limit 5
  uv run python parquet_reader.py --dataset indiccorp --subfolder hi --sample 10
  uv run python parquet_reader.py --dataset ncert --subfolder _all --schema
  
  # Convert parquet to JSON file
  uv run python parquet_reader.py --dataset sangraha --subfolder hin --format json --export output.json
        """
    )
    parser.add_argument("--dataset", required=True, choices=["dolma", "sangraha", "indiccorp", "ncert"],
                        help="Dataset name")
    parser.add_argument("--subfolder", required=True,
                        help="Subfolder (source for dolma, language for sangraha/indiccorp, '_all' for ncert)")
    parser.add_argument("--format", dest="output_format", default="table", choices=["table", "json", "jsonl"],
                        help="Output format: table (default), json, or jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to display/export")
    parser.add_argument("--sample", type=int, default=None,
                        help="Show N random sample records")
    parser.add_argument("--schema", action="store_true",
                        help="Show schema information")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics only")
    parser.add_argument("--export", type=str, default=None,
                        help="Export to file (format determined by --format: json or jsonl)")
    
    args = parser.parse_args()
    
    # Always read from parquet
    reader = DatasetReader(args.dataset, args.subfolder, "parquet")
    
    # Check if data exists
    if not reader.get_files():
        print(f"No parquet files found at: {reader.base_path}")
        print(f"Run: uv run python main.py --dataset {args.dataset} --scope test --format parquet")
        return
    
    print(reader)
    print()
    
    # Show stats
    if args.stats:
        stats = reader.get_stats()
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Show schema
    if args.schema:
        schema = reader.get_schema()
        print("Schema:")
        for col, dtype in schema.get("schema", {}).items():
            print(f"  {col}: {dtype}")
        print(f"\nRows in first file: {schema.get('rows', 0)}")
        return
    
    # Export to file
    if args.export:
        if args.output_format == "json":
            reader.to_json(args.export, limit=args.limit)
        elif args.output_format == "jsonl":
            reader.to_jsonl(args.export, limit=args.limit)
        else:
            print("Error: --export requires --format json or --format jsonl")
        return
    
    # Output as JSON to stdout
    if args.output_format == "json":
        limit = args.limit or 5
        print(f"Records as JSON (showing {limit}):")
        json_str = reader.to_json(limit=limit)
        print(json_str)
        return
    
    # Output as JSONL to stdout
    if args.output_format == "jsonl":
        limit = args.limit or 5
        print(f"Records as JSONL (showing {limit}):")
        jsonl_str = reader.to_jsonl(limit=limit)
        print(jsonl_str)
        return
    
    # Show sample (table format)
    if args.sample:
        print(f"Sample ({args.sample} records):")
        print("-" * 80)
        sample_df = reader.sample(args.sample)
        for idx, row in sample_df.iterrows():
            print(f"ID: {row['id']}")
            print(f"Hash: {row.get('hash', 'N/A')[:16]}...")
            print(f"Language: {row['language']}")
            text_preview = row['text'][:200].replace('\n', ' ')
            print(f"Text: {text_preview}...")
            print("-" * 80)
        return
    
    # Default: show records in table format
    limit = args.limit or 5
    print(f"Records (showing {limit}):")
    print("-" * 80)
    for i, record in enumerate(reader.iter_records(limit=limit)):
        print(f"[{i+1}] ID: {record['id']}")
        print(f"    Hash: {record.get('hash', 'N/A')[:16]}...")
        print(f"    Dataset: {record['dataset']}, Language: {record['language']}")
        text_preview = record['text'][:200].replace('\n', ' ')
        print(f"    Text: {text_preview}...")
        print()


if __name__ == "__main__":
    main()
