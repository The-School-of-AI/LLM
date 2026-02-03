import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

# ------------------------
# CONFIGURATION
# ------------------------
INPUT_BUCKET = "s3://bucket/enriched"
INDEX_BUCKET = f"{INPUT_BUCKET}/_index/global_index.parquet"
BATCH_SIZE = 10000

# ------------------------
# INITIALIZE
# ------------------------
fs = s3fs.S3FileSystem()

# List all enriched parquet files
files = fs.glob(f"{INPUT_BUCKET}/*.parquet")
if not files:
    raise RuntimeError(f"No parquet files found in {INPUT_BUCKET}")

print(f"Found {len(files)} files to index")

# ------------------------
# STREAMING BUILD
# ------------------------
# We'll append batches to a Parquet file incrementally using pyarrow's ParquetWriter
first_batch = True
writer = None

for f in files:
    print(f"Processing {f}")
    pf = pq.ParquetFile(f, filesystem=fs)

    row_offset = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["id", "curriculum_band"]):
        ids = batch.column("id").to_pylist()
        bands = batch.column("curriculum_band").to_pylist()
        rows = []

        for i in range(len(ids)):
            rows.append(
                {
                    "id": ids[i],
                    "band": bands[i],
                    "file": f,
                    "row": row_offset + i,
                }
            )
        row_offset += len(ids)

        table = pa.Table.from_pylist(rows)

        # Initialize ParquetWriter once with schema
        if first_batch:
            with fs.open(INDEX_BUCKET, "wb") as f_s3:
                writer = pq.ParquetWriter(f_s3, table.schema)
            first_batch = False

        # Write batch to S3
        with fs.open(INDEX_BUCKET, "ab") as f_s3:
            pq.write_table(table, f_s3)

print(f"Global index written to {INDEX_BUCKET}")
