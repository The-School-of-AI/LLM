import ray
import s3fs
from tqdm import tqdm

from curriculum_tags.engine import CurriculumTagger

# ------------------------
# CONFIGURATION
# ------------------------
INPUT_S3_PREFIX = "s3://my-bucket/dolma_parquet/"
OUTPUT_S3_PREFIX = "s3://my-bucket/dolma_enriched/"
CURRICULUM_YAML = "/home/ubuntu/curriculum.yaml"
BATCH_SIZE = 10000
NUM_CPUS = 8  # parallelism = number of files processed at once
MAX_INFLIGHT = NUM_CPUS * 2

# ------------------------
# INITIALIZE
# ------------------------
ray.init(num_cpus=NUM_CPUS)


# ------------------------
# PROCESS ONE FILE
# ------------------------
@ray.remote(num_cpus=1)
def process_s3_file(file_path: str) -> dict:
    fs = s3fs.S3FileSystem()
    tagger = CurriculumTagger(CURRICULUM_YAML)

    relative_path = file_path.replace(INPUT_S3_PREFIX, "")
    output_file = OUTPUT_S3_PREFIX + relative_path

    try:
        stats = tagger.process_parquet_s3(
            file_path,
            output_file,
            filesystem=fs,
            batch_size=BATCH_SIZE,
        )

        return {
            "input_file": file_path,
            "output_file": output_file,
            "status": "success",
            **stats,
        }

    except Exception as e:
        return {
            "input_file": file_path,
            "output_file": output_file,
            "status": "failed",
            "error": str(e),
        }


def process_s3_bucket():
    # List all input Parquet files on S3
    fs = s3fs.S3FileSystem()
    # all_inputs = fs.ls(INPUT_S3_PREFIX)
    all_inputs = [f for f in fs.ls(INPUT_S3_PREFIX) if f.endswith(".parquet")]

    all_files = []
    for f in all_inputs:
        rel = f.replace(INPUT_S3_PREFIX, "")
        out = OUTPUT_S3_PREFIX + rel
        if fs.exists(out):
            print(f"[SKIP] Already exists: {rel}")
        else:
            all_files.append(f)

    print(f"Found {len(all_files)} files to process.")

    pending = []
    failures = []
    file_iter = iter(all_files)

    # Prime the pipeline
    for _ in range(min(MAX_INFLIGHT, len(all_files))):
        try:
            f = next(file_iter)
        except StopIteration:
            break
        pending.append(process_s3_file.remote(f))

    pbar = tqdm(total=len(all_files), desc="Processing files")

    while pending:
        done, pending = ray.wait(pending, num_returns=1)

        res = ray.get(done[0])

        if res and res["status"] == "failed":
            failures.append(res)

        pbar.update(1)

        # Refill pipeline
        try:
            f = next(file_iter)
            pending.append(process_s3_file.remote(f))
        except StopIteration:
            pass

    print(f"Failures: {len(failures)}")
    if failures:
        print("Example failure:", failures[0])
    pbar.close()
    ray.shutdown()


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    process_s3_bucket()
