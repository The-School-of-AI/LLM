#!/bin/bash
set -e

DOLMA_VERSION="v1_7"
SAMPLE_COUNT=3

WORKDIR="/mnt/dolma_tmp"
mkdir -p "$WORKDIR"

S3_BUCKET="s3://upload-dataset"
S3_PREFIX="dolma_sample/${DOLMA_VERSION}/"

# Install tools
sudo dnf install -y git wget awscli

# Clone repo
git clone https://huggingface.co/datasets/allenai/dolma || true

URL_FILE="dolma/urls/${DOLMA_VERSION}.txt"

echo "Downloading first ${SAMPLE_COUNT} shards..."

head -n ${SAMPLE_COUNT} "$URL_FILE" | while read -r url; do

  filename=$(basename "$url")

  echo "Downloading: $filename"

  # Resume-enabled download
  wget -c -P "$WORKDIR" "$url"

  echo "Uploading to S3..."
  aws s3 cp "$WORKDIR/$filename" \
           "${S3_BUCKET}/${S3_PREFIX}${filename}"

  echo "Deleting local file..."
  rm -f "$WORKDIR/$filename"

done

echo "Sample test complete!"
