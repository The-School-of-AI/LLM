# Data Mount Points

Training configs use this directory for local shard and auxiliary data paths.

```text
d1_shards/              default shard root for 2B, 5B, and 9B configs
curriculum_test_shards/ synthetic shards created by create_curriculum_test_shards.py
training_shards_8k/     large shard root for long-context TQP configs
golden_proxy.pt         optional OPUS proxy tensor
```

You can replace these directories with symlinks or update the YAML configs to
point at your own mounted storage.
