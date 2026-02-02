import pyarrow as pa
import pyarrow.parquet as pq
import ray
import s3fs
from tqdm import tqdm  # optional for progress bar

from curriculum_tags.engine import CurriculumTagger

# ------------------------
# CONFIGURATION
# ------------------------
INPUT_S3_PREFIX = "s3://my-bucket/dolma_parquet/"
OUTPUT_S3_PREFIX = "s3://my-bucket/dolma_enriched/"
CURRICULUM_YAML = "/home/ubuntu/curriculum.yaml"
BATCH_SIZE = 10000
NUM_CPUS = 8  # parallelism = number of files processed at once
RESUME = True  # skip files that already exist in S3

# ------------------------
# INITIALIZE
# ------------------------
ray.init(num_cpus=NUM_CPUS)
fs = s3fs.S3FileSystem()
tagger = CurriculumTagger(CURRICULUM_YAML)


# ------------------------
# PROCESS ONE FILE
# ------------------------
def process_s3_file(file_path: str) -> str:
    """Process a single Parquet file from S3, add curriculum tags, and write back."""
    relative_path = file_path.replace(INPUT_S3_PREFIX, "")
    output_file = OUTPUT_S3_PREFIX + relative_path

    # Skip if already processed
    if RESUME and fs.exists(output_file):
        print(f"[SKIP] Already processed: {relative_path}")
        return output_file

    try:
        pf = pq.ParquetFile(file_path, filesystem=fs)
        output_batches = []

        # Process in batches
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            records = batch.to_pylist()

            # # Optional: flatten nested structs to avoid Arrow errors
            # for rec in records:
            #     for col in ["metadata", "source"]:
            #         if col in rec:
            #             rec[col] = str(rec[col]) if rec[col] is not None else "{}"

            # Enrich each record with CurriculumTagger
            tagged_records = [tagger.tag_sample(r) for r in records]

            # Convert batch back to Arrow
            output_batches.append(pa.Table.from_pylist(tagged_records))

        # Concatenate all batches
        output_table = pa.concat_tables(output_batches)

        # Atomic write: write to tmp file first
        tmp_output = output_file + ".tmp"
        with fs.open(tmp_output, "wb") as f:
            pq.write_table(output_table, f)

        # Move to final output path
        fs.mv(tmp_output, output_file)

        print(f"[DONE] Processed: {relative_path}")
        return output_file

    except Exception as e:
        print(f"[ERROR] Failed {relative_path}: {e}")
        return None


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    # List all input Parquet files on S3
    all_files = fs.ls(INPUT_S3_PREFIX)
    print(f"Found {len(all_files)} files to process.")

    # Submit tasks to Ray
    futures = [ray.remote(process_s3_file).remote(f) for f in all_files]

    results = []
    for f in tqdm(futures, desc="Processing files"):
        r = ray.get(f)
        if r is not None:
            results.append(r)

    print("Processing completed. Enriched files:")
    for f in results:
        print(f)
