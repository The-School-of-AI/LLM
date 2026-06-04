# Runtime Hot Configuration

Some MoE training controls are read at runtime rather than fixed only in YAML.
Use these controls when reproducing the LightningLM MoE training recipe or when
continuing from a checkpoint that expects the same router-balance behavior.

## 120B Router Balance Controls

The 120B stage uses loss-free expert balancing: router biases are adjusted by a
controller instead of adding a differentiable load-balancing loss to the training
objective.

Recommended starting values:

```json
{
  "MOE_BIAS_GAMMA": "5e-5",
  "MOE_EXPERT_CAP_LO": "0.0005",
  "MOE_EXPERT_CAP_HI": "0.004",
  "MOE_W_Z": "0"
}
```

`MOE_W_Z=0` means the z-loss can be logged for inspection but does not
contribute to the training objective.

## Applying A Hot Config

The runtime reads the hot config from:

```text
/tmp/moe_hotconfig.json
```

Write the JSON before launching training, or update it at a checkpoint boundary
when continuing a run.

Example:

```bash
cat > /tmp/moe_hotconfig.json <<'JSON'
{
  "MOE_BIAS_GAMMA": "5e-5",
  "MOE_EXPERT_CAP_LO": "0.0005",
  "MOE_EXPERT_CAP_HI": "0.004",
  "MOE_W_Z": "0"
}
JSON
```

## 9B AON Rampdown

The 9B stage supports operator-side AON mix adjustment during long runs. For a
new run, prefer the YAML data/curriculum settings as the source of truth. For a
continuation from an existing checkpoint, apply the same hot-config values used
for that continuation before resuming.

## Practical Rule

For new users training their own model, start from the config values in
`configs/` and use hot config only when intentionally changing router-balance or
AON behavior mid-run. For reproducing a particular run segment, write the
expected hot config before resuming from that segment's checkpoint.
