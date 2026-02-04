from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

# ------------------------
# CONFIGURATION
# ------------------------
INPUT_BUCKET = "s3://bucket/enriched"
INDEX_FOLDER = f"{INPUT_BUCKET}/_index/"
BATCH_SIZE = 10000

fs = s3fs.S3FileSystem()

files = [
    f
    for f in fs.glob(f"{INPUT_BUCKET}/*.parquet")
    if not f.endswith(".metadata.parquet")
]

if not files:
    raise RuntimeError(f"No parquet files found in {INPUT_BUCKET}")

print(f"Found {len(files)} files to index")

# Make sure index folder exists
if not fs.exists(INDEX_FOLDER):
    fs.mkdir(INDEX_FOLDER)

# ------------------------
# STEP 1: Build per-file mini-index
# ------------------------
mini_index_paths = []

for f in files:
    print(f"Processing {f}")
    pf = pq.ParquetFile(f, filesystem=fs)

    row_offset = 0
    mini_rows = []

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE, columns=["id", "curriculum_tags"]
    ):
        ids = batch.column("id").to_pylist()
        bands = (
            batch.column("curriculum_tags")
            .field("band_assignment")
            .field("band")
            .to_pylist()
        )

        mini_rows.extend(
            [
                {"id": ids[i], "band": bands[i], "file": f, "row": row_offset + i}
                for i in range(len(ids))
            ]
        )
        row_offset += len(ids)

    # Write mini-index for this file
    table = pa.Table.from_pylist(mini_rows)
    mini_index_file = f"{INDEX_FOLDER}{Path(f).stem}_index.parquet"
    with fs.open(mini_index_file, "wb") as f_s3:
        pq.write_table(table, f_s3)
    mini_index_paths.append(mini_index_file)
    print(f"Mini-index written: {mini_index_file}")

# ------------------------
# STEP 2: Merge mini-indexes into global_index.parquet
# ------------------------
global_index_path = f"{INDEX_FOLDER}global_index.parquet"

# Use streaming concat to avoid loading all data in memory
tables = []
for idx_file in mini_index_paths:
    table = pq.read_table(idx_file, filesystem=fs)
    tables.append(table)

# Concatenate all tables
global_table = pa.concat_tables(tables)

# Write final global index
with fs.open(global_index_path, "wb") as f_s3:
    pq.write_table(global_table, f_s3)

print(f"Global index written to {global_index_path}")
