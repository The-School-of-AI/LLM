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
- ** Prerequisites:**
   Install the required packages for environment reproducibility:

  ```bash
     pip install uv wget boto3
  ```
-** Logging:**
  Logs and audit trails are stored in the `/data/dolma/logs` directory. Check this location for detailed logs.

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

## Handling Failures and Resuming Downloads

### Per-File Logging and Job Structure
- Each Dolma download job is triggered per dataset file (one file per job), and logs are generated for each file indicating whether the download was successful or failed.
- Logs and audit trails are stored in the `/data/dolma/logs` directory. Check this location for detailed logs and status of each file.

### Resuming After Failures
- If a download fails (e.g., due to network or disk issues), the failed file's URL will be recorded in the logs.
- To resume, create a new subset file (e.g., `dolma_urls_failed.txt`) containing only the URLs of the files that failed to download.
- Rerun the batch script with this new subset file:
  ```bash
  ./download_dolma_all_dataset_batch.sh dolma_urls_failed.txt ./datasets/
  ```
- This will attempt to download only the remaining files.

### Parallelization and Resource Considerations
- You can trigger multiple download jobs in parallel to speed up the process.
- The number of concurrent jobs you can run depends on your machine's CPU, memory, disk I/O, and especially network bandwidth.
- For very large datasets (like Dolma), it is recommended to:
  - Monitor system resource usage (CPU, RAM, disk, network) during downloads.
  - Start with a small number of parallel jobs (e.g., 2-4) and increase as capacity allows.
  - Use a robust cloud instance (e.g., AWS EC2 with high network throughput and sufficient EBS storage).
- Always ensure you do not exceed your storage or bandwidth limits to avoid partial/corrupted downloads.

For questions or issues, contact the project maintainers.
