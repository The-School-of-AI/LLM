#!/bin/bash

# Usage: ./download_dumbo.sh /path/to/url_file.txt


URL_FILE="$1"
DEST_DIR="/data/dolma/dolma_dataset/"
BASE_DIR="/data/dolma/logs"

if [ -z "$URL_FILE" ]; then
    echo "Usage: $0 /path/to/url_file.txt"
    exit 1
fi

if [ ! -f "$URL_FILE" ]; then
    echo "Error: URL file '$URL_FILE' does not exist."
    exit 1
fi

# Get prefix from input file name (strip path and extension)
BASE_NAME=$(basename "$URL_FILE")
PREFIX="${BASE_NAME%%.*}"

# Ensure BASE_DIR exists
mkdir -p "$BASE_DIR"


# Create subdirectory under DEST_DIR using PREFIX
SUB_DEST_DIR="$DEST_DIR$PREFIX"
mkdir -p "$SUB_DEST_DIR"

LOG_FILE="$BASE_DIR/${PREFIX}_download_status.log"
ERR_FILE="$BASE_DIR/${PREFIX}_download_errors.log"

# Clear previous logs

> "$LOG_FILE"
> "$ERR_FILE"


echo "Starting downloads from $URL_FILE..." | tee -a "$LOG_FILE"

# Export variables for use in subshell
export SUB_DEST_DIR LOG_FILE ERR_FILE

cat "$URL_FILE" | xargs -n 1 -P 2 -I {} bash -c '
    url="{}"
    fname=$(basename "$url")
    wget -q -c --tries=3 --waitretry=5 -P "$SUB_DEST_DIR" "$url"
    status=$?
    if [ $status -eq 0 ]; then
        echo "SUCCESS: $fname" >> "$LOG_FILE"
    else
        echo "ERROR: $fname (URL: $url)" >> "$ERR_FILE"
    fi
' &

BG_PID=$!
echo "Download process started in background (PID: $BG_PID)."

wait $BG_PID

echo "All downloads finished." | tee -a "$LOG_FILE"
echo "Summary:"
echo "Successes: $(wc -l < "$LOG_FILE")"
echo "Errors: $(wc -l < "$ERR_FILE")"

