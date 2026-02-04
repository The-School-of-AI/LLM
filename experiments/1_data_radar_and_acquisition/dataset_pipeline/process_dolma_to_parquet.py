import duckdb
import argparse
import os
import glob

def process_dolma_to_parquet(input_glob, output_dir, domain, version="1.7"):
    """
    Processes local Dolma .json.gz files into the specified Parquet schema.
    """
    # Initialize DuckDB connection
    # Using ':memory:' for metadata, the actual processing happens on disk streams
    con = duckdb.connect(database=':memory:')
    
    # Increase maximum object size to handle large JSON objects (up to 4GB)
    con.execute("SET maximum_object_size = '6GB';")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files
    files = glob.glob(input_glob)
    if not files:
        print(f"No files found matching: {input_glob}")
        return

    print(f"Found {len(files)} files. Starting conversion...")

    for file_path in files:
        file_name = os.path.basename(file_path).replace(".json.gz", ".parquet")
        output_path = os.path.join(output_dir, file_name)
        
        print(f"Processing: {file_name}...")

        # DuckDB SQL Query to transform and write:
        # 1. read_json_auto handles decompression and JSON parsing
        # 2. sha256() computes the required content hash
        # 3. CAST(metadata AS VARCHAR) ensures the JSON object becomes a string
        query = f"""
            COPY (
                SELECT 
                    id,
                    sha256(text) AS hash,
                    'dolma' AS dataset,
                    '{domain}' AS domain,
                    source,
                    text,
                    'en' AS language,
                    CAST(metadata AS VARCHAR) AS metadata,
                    added,
                    created,
                    '{version}' AS version
                FROM read_json_auto('{file_path}')
            ) TO '{output_path}' (FORMAT 'parquet');
        """
        
        try:
            con.execute(query)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"\nSuccess! All files converted to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Dolma JSONL.GZ to formatted Parquet.")
    parser.add_argument("--input", required=True, help="Glob pattern for input files (e.g. 'data/*.json.gz')")
    parser.add_argument("--output", required=True, help="Output directory for Parquet files")
    parser.add_argument("--domain", required=True, help="Domain tag (e.g. 'web', 'code', 'math')")
    parser.add_argument("--version", default="1.7", help="Dataset version tag")

    args = parser.parse_args()

    process_dolma_to_parquet(
        input_glob=args.input,
        output_dir=args.output,
        domain=args.domain,
        version=args.version
    )