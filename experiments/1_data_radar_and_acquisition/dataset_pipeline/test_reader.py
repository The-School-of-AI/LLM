"""Test script for DatasetReader."""

from reader import DatasetReader


def test_reader():
    print("=" * 60)
    print("Testing DatasetReader")
    print("=" * 60)

    # Test JSON reader
    print("\n=== JSON Reader ===")
    json_reader = DatasetReader("sangraha", "hin", "json")
    print(json_reader)
    print("Dataset name:", json_reader.name)
    print("Files:", json_reader.get_files())
    print("Stats:", json_reader.get_stats())

    # Test Parquet reader
    print("\n=== Parquet Reader ===")
    parquet_reader = DatasetReader("sangraha", "hin", "parquet")
    print(parquet_reader)
    print("Dataset name:", parquet_reader.name)
    print("Files:", parquet_reader.get_files())
    print("Stats:", parquet_reader.get_stats())
    print("Schema:", parquet_reader.get_schema())

    # Test sample
    print("\n=== Sample Records ===")
    sample = parquet_reader.sample(3)
    print(sample[["id", "language", "metadata"]])

    # Test read with limit
    print("\n=== Read with Limit (5 records) ===")
    df_limited = parquet_reader.read(limit=5)
    print(f"Requested 5, got {len(df_limited)} records")
    print(df_limited[["id", "language"]].head())

    # Test iter_records with limit
    print("\n=== Iterate Records with Limit (3 records) ===")
    for i, record in enumerate(parquet_reader.iter_records(limit=3)):
        print(f"Record {i}: id={record['id'][:20]}..., lang={record['language']}")

    # Test read_all
    print("\n=== Read All ===")
    df = parquet_reader.read_all()
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Test non-existent dataset
    print("\n=== Non-existent Dataset ===")
    empty_reader = DatasetReader("nonexistent", "test", "parquet")
    print(empty_reader)
    print("Dataset name:", empty_reader.name)
    print("Files:", empty_reader.get_files())

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_reader()
