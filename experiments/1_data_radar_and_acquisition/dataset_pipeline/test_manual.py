from datasets import load_dataset
import requests

def test_load_dolma_json():
    print("Testing manual load of Dolma v1.7 JSON URLs...")
    # List of some URLs for v1.7
    urls = [
        "https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz",
        "https://olmo-data.org/dolma-v1_7/books/books-0001.json.gz"
    ]
    try:
        dataset = load_dataset("json", data_files=urls, split="train", streaming=True)
        for i, example in enumerate(dataset.take(1)):
            print(f"Success! {example['text'][:100]}...")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_load_dolma_json()
