import argparse
import sys
import os

# Add src to path if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from coreset_engine.selection.builder import CoresetBuilder
except ImportError as e:
    print(f"ImportError: {e}")
    # Fallback for direct running
    sys.path.append('src')
    from coreset_engine.selection.builder import CoresetBuilder

def main():
    parser = argparse.ArgumentParser(description="Team 3 Coreset Builder")
    parser.add_argument("--config", required=True, help="Path to curriculum.yaml")
    parser.add_argument("--data", required=True, help="Path to input data directory (JSONL/Parquet)")
    parser.add_argument("--output", required=True, help="Path to output directory for manifests")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found {args.config}")
        sys.exit(1)
        
    builder = CoresetBuilder(args.config, args.data, args.output)
    builder.build()

if __name__ == "__main__":
    main()
