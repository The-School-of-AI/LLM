import pyarrow as pa
import xxhash


def deterministic_shuffle(table: pa.Table, seed: int):
    ids = table["id"].to_pylist()

    hashes = [xxhash.xxh64(str(i), seed=seed).intdigest() for i in ids]

    hash_array = pa.array(hashes, type=pa.uint64())
    table = table.append_column("_hash", hash_array)

    table = table.sort_by([("_hash", "ascending")])
    table = table.drop(["_hash"])

    return table
