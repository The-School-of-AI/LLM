import pandas as pd
import sys
import os

def convert_jsonl_to_parquet(input_file, output_file):
    print(f"Reading {input_file}...")
    try:
        # Read JSONL file into pandas DataFrame
        df = pd.read_json(input_file, lines=True)
        
        print(f"Converting {len(df)} rows to Parquet...")
        # Save as Parquet
        df.to_parquet(output_file, engine='pyarrow', index=False)
        print(f"Successfully saved to {output_file}")
        
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl_parquet.py <input_jsonl> <output_parquet>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_jsonl_to_parquet(input_path, output_path)
