# LLM

Lightning Language Models

## Dolma Custom URL File Usage

To run Dolma downloads with a custom set of data files, provide a plain text file containing URLs (one per line) using the `--dolma-urls-file` argument:

```
python main.py --dataset dolma --dolma-urls-file /path/to/your/urls.txt --scope test --format json
```

**The file must be a plain text file with one URL per line** (not a list of file paths or Python list). Example:

```
https://huggingface.co/datasets/allenai/dolma/resolve/main/data/v1_7/file1.json.gz
https://huggingface.co/datasets/allenai/dolma/resolve/main/data/v1_7/file2.json.gz
...etc
```

If not provided, the default Dolma v1.7 file list from Hugging Face will be used.

## Contribution Guidelines

See [contribution guidelines](https://github.com/The-School-of-AI/LLM/tree/main/experiments/19_reproducibility_provenance_and_experiment_tracking/contribution.md)

## Rebase with Staging

See [rebase with staging](https://github.com/The-School-of-AI/LLM/tree/main/experiments/19_reproducibility_provenance_and_experiment_tracking/rebase_with_stage.md)
