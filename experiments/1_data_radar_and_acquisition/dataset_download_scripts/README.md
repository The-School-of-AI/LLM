# Dataset Download Scripts Documentation

This folder contains scripts for downloading Dolma datasets to your local device.

## Scripts

### 1. download_dolma_dataset.sh
- **Purpose:** Downloads a dataset file from a specified URL and saves it locally.
- **Usage:**
  ```
  ./download_dolma_dataset.sh <dataset_url> <output_path>
  ```
  - `<dataset_url>`: The URL of the dataset file to download.
  - `<output_path>`: The local path where the file will be saved.

### 2. download_dolma_all_dataset_batch.sh
- **Purpose:** Batch script to download multiple datasets by iterating over a list of dataset URLs and calling `download_dolma_dataset.sh` for each.
- **Usage:**
  ```
  ./download_dolma_all_dataset_batch.sh <url_list_file> <output_dir>
  ```
  - `<url_list_file>`: A text file containing dataset URLs, one per line.
  - `<output_dir>`: The directory where all downloaded files will be saved.

## Example Workflow
1. Prepare a text file (e.g., `dolma_urls.txt`) with each dataset URL on a new line.
2. Run the batch script:
   ```
   ./download_dolma_all_dataset_batch.sh dolma_urls.txt ./datasets/
   ```
   This will download all datasets listed in `dolma_urls.txt` to the `./datasets/` directory.

## Notes
- Ensure both scripts have execute permissions (`chmod +x <script_name>`).
- The batch script relies on `download_dolma_dataset.sh` being present in the same directory.
- Download progress and errors will be shown in the terminal.

For questions or issues, contact the project maintainers.
