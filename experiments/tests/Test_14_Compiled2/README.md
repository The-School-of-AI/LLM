# Test_14_gsa_only_liger_kernels_1000steps

## Objective
GSA-only reversible model (no DeltaNet): all 8 layers use GSA only. Liger kernels are used for RoPE, SwiGLU MLP, and fused linear+cross-entropy (CE). Training uses the fused CE path when `use_fused_ce: true`. Kronecker embeddings; 1000 steps.

## Run
```bash
cd "experiments/tests/Test_14_gsa_only_liger_kernels_1000steps"
./run.sh
```

Optional:
```bash
NUM_GPUS=4 ./run.sh
```

Force regenerate init model:
```bash
FORCE_REWRITE_INIT=1 ./run.sh
```

## Fixed controls
- model_variant: `reversible`
- embedding_type: `kronecker`
- max_train_steps: `1000`
- use_fused_ce: `true`
- log_interval: `1`
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with the reversible model only. No DeltaNet; Liger RoPE, Liger SwiGLU MLP, and Liger fused CE are used; fused CE is used in the training loop.

## TrainingOps integration (observability)

This test folder vendors `components/` for P12 observability. The training loop initializes `TrainingOps` automatically and logs:

- **Events**: stage transitions, checkpoints
- **Metrics**: per-step loss, throughput, learning rate
- **System metrics**: emitted by a background collector (when ops is active)

### Required environment variables (remote ClickHouse)

Set the following before running (export in shell or via user-data on instances):

- `CLICKHOUSE_HTTPS_ENDPOINT` or `CLICKHOUSE_HTTP_ENDPOINT` or `CLICKHOUSE_ENDPOINT`
- `CLICKHOUSE_USER`
- `CLICKHOUSE_PASSWORD`
- `CLICKHOUSE_CA_CERT` (path to CA for TLS endpoints; optional if HTTP)
- `VECTOR_SERVICE_NAME` (default `p12-vector.service`; set to `t12-vector.service` if using the sidecar in this repo)
- `SKIP_VECTOR_CHECK` (set `0` when Vector sidecar is running; `1` to skip preflight during local debug)

Default log directory for Vector tailing: `/tmp/training_logs`.

### Quickstart (single box or cluster)

```bash
export CLICKHOUSE_HTTPS_ENDPOINT="https://<your-ch-host>:8443"
export CLICKHOUSE_USER="p12_writer"
export CLICKHOUSE_PASSWORD="<secret>"
export CLICKHOUSE_CA_CERT="/etc/t12/ca.crt"   # if present
export VECTOR_SERVICE_NAME="t12-vector.service"
export SKIP_VECTOR_CHECK=0                    # ensure Vector is running

cd experiments/tests/Test_14_Compiled2
./run.sh
```

## Spot instances: bootstrap & automation

Training typically runs on EC2 Spot. Use a Launch Template with user data to prepare Vector + env for the training process.

### Sidecar user-data script

Use `code/components/sidecar_agent/userdata_vector.sh` as EC2 user data. It:

- Installs Vector
- Downloads TLS CA and Vector config
- Assumes a cross-account role to read a Secrets Manager secret `t12/clickhouse`
- Writes `/etc/t12/vector.env` and copies to `/home/ubuntu/.t12.env`
- Starts `t12-vector.service` (systemd)
- Installs a CloudWatch healthcheck cron

Edit these variables near the top of the script before use:

- `T12_CONFIG_BUCKET` (S3 bucket for CA/config)
- `AWS_REGION`
- `PREFIX` (unique resource prefix, if following the infra scripts)
- `SECRETS_ROLE_ARN` (role permitted to read `t12/clickhouse`)

### Minimal run.sh change (recommended)

Source the env created by user-data so training inherits ClickHouse creds:

```bash
# At the top of run.sh before invoking deepspeed
set -a
[ -f "$HOME/.t12.env" ] && source "$HOME/.t12.env"
set +a
```

### Networking & IAM checklist

- Security Group: outbound HTTPS (443) to S3/Secrets Manager/STS/CloudWatch, and TCP 8443 (or your CH port) to ClickHouse
- Instance profile: permissions to assume `SECRETS_ROLE_ARN` and access required AWS APIs
- Secrets Manager: secret `t12/clickhouse` containing `endpoint` and writer credentials
- TLS: place CA cert at `/etc/t12/ca.crt` (or update `CLICKHOUSE_CA_CERT` path)

### Verification on instance

- `systemctl status t12-vector` → active
- `curl -sk --cacert /etc/t12/ca.crt "$CLICKHOUSE_HTTPS_ENDPOINT/?query=SELECT+1" -H "X-ClickHouse-User: $CLICKHOUSE_USER" -H "X-ClickHouse-Key: $CLICKHOUSE_PASSWORD"`
- `curl -s http://localhost:8686/health` (Vector health)
- Start training; verify logs arrive in ClickHouse

### Spot preemption (next step)

Add a termination notice handler (via systemd or IMDS watch) to:

- Emit a `checkpoint_saving` event
- Save a final checkpoint
- Flush/shutdown Vector gracefully
