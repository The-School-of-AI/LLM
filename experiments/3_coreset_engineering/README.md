# Team 3: Coreset Engineering Experiment

## Overview
This directory contains the engineering pipeline for creating **Stage-Specific Coresets** for LLM Training.
The pipeline absorbs raw data chunks, scores them for difficulty, removes duplicates, and samples them into curriculum-aligned stages (1B -> 3B -> 8B -> 70B).

**Key Features:**
*   **Production Curriculum Support**: Parses `curriculum.yaml` to enforce specific Profiles, Modalities (Code, CoT, etc.), and Difficulty Bands (B0-B5).
*   **Stratified Sampling**: Ensures strict adherence to `band_weights` and `modality_weights`.
*   **Reproducibility**: Deterministic pipeline with verifiable manifest outputs.

## Pipeline Workflow

```text
+----------------+      +-----------+       +-------+
| Raw Data Input | ---> | Ingestion | --->  | Dedup |
+----------------+      +-----------+       +-------+
                                                |
                                           (Unique Only)
                                                |
                                                v
                                       +-------------------+
                                      # Team 3: Coreset Engineering

                                      This folder contains the production coreset engineering pipeline, configurations, scripts, tests, and reproducibility artifacts for Module 3.

                                      Key consolidated documentation:

                                      - `REPRODUCIBILITY_POLICY.md` — Reproducibility requirements and policies (seed policy, manifest fingerprints, audit trail).
                                      - `IMPLEMENTATION_GUIDE.md` — Implementation notes and quick-start examples.

                                      Quick links:

                                      - Source: `src/coreset_engine/`
                                      - Configs: `configs/` (includes `curriculum.yaml`)
                                      - Scripts: `scripts/` (builder, mock data, audits)
                                      - Tests: `tests/` (consolidated module tests and instructions)

                                      Running the module (recommended via `uv`):

                                      ```powershell
                                      uv sync
                                      uv run experiments/3_coreset_engineering/scripts/coreset_builder.py \
                                        --config experiments/3_coreset_engineering/configs/curriculum.yaml \
                                        --data experiments/3_coreset_engineering/data/input \
                                        --output experiments/3_coreset_engineering/output/manifests
                                      ```

                                      Validate generated manifests (local/CI):

                                      ```powershell
                                      python experiments/3_coreset_engineering/scripts/validate_manifests.py \
                                        --manifest-dir experiments/3_coreset_engineering/output/manifests
                                      ```

                                      Tests (module-level):

                                      ```powershell
                                      uv run pytest experiments/3_coreset_engineering/tests -v
                                      ```

                                      Notes:
                                      - The detailed pipeline design and examples (previously duplicated in several READMEs) have been consolidated here and into `REPRODUCIBILITY_POLICY.md` and `IMPLEMENTATION_GUIDE.md`.
                                      - If you need historical drafts or expanded examples, see `IMPLEMENTATION_GUIDE.md` and `REPRODUCIBILITY_POLICY.md`.

                                      CI / Running in CI
                                      ------------------

                                      Minimum environment and setup:

                                      - **Python**: 3.12 (CI and local runs should use the same version).
                                      - **uv**: recommended for reproducible installs. If `uv` is not available, install via `pip install uv` or use `python -m pip install -r requirements.txt`.
                                      - Use locked dependencies in CI: `uv sync --frozen` (enforces `uv.lock`).

                                      Required environment variables / GitHub Secrets (set these in repo Settings → Secrets):

                                      - `AWS_ACCESS_KEY_ID` (if uploading/testing with S3)
                                      - `AWS_SECRET_ACCESS_KEY`
                                      - `AWS_DEFAULT_REGION`
                                      - `S3_TEST_BUCKET` (optional, used by some integration tests)

                                      Deterministic CI flags (recommended):

                                      ```yaml
                                      env:
                                        PYTHONHASHSEED: '0'
                                        PYTHONDONTWRITEBYTECODE: '1'
                                      ```

                                      Validate manifests in CI
                                      ------------------------

                                      Use the included validator to fail CI on manifest issues. The validator returns a non-zero exit code on validation failure so CI jobs fail fast:

                                      ```powershell
                                      python experiments/3_coreset_engineering/scripts/validate_manifests.py --manifest-dir output/manifests
                                      ```

                                      Coverage and artifacts
                                      ----------------------

                                      - Run tests with coverage and upload artifacts from CI (example given in Actions snippet).
                                      - Consider caching pip/uv artifacts in CI to speed up installs.

                                      Recommended GitHub Actions snippet
                                      ---------------------------------

                                      Use the snippet below as a starting point for a `coreset-deploy.yml` job (copy into `.github/workflows/`):

                                      ```yaml
                                      name: Coreset CI

                                      on: [push, pull_request]

                                      jobs:
                                        test:
                                          runs-on: ubuntu-latest
                                          env:
                                            PYTHONHASHSEED: '0'
                                            PYTHONDONTWRITEBYTECODE: '1'
                                          steps:
                                            - uses: actions/checkout@v4
                                            - name: Setup Python
                                              uses: actions/setup-python@v4
                                              with:
                                                python-version: '3.12'
                                            - name: Cache pip
                                              uses: actions/cache@v4
                                              with:
                                                path: ~/.cache/pip
                                                key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
                                            - name: Install uv & sync
                                              run: |
                                                python -m pip install --upgrade pip
                                                pip install uv || true
                                                uv sync --frozen
                                            - name: Run tests
                                              run: uv run pytest experiments/3_coreset_engineering/tests -v --maxfail=1
                                            - name: Validate manifests
                                              run: |
                                                python experiments/3_coreset_engineering/scripts/validate_manifests.py --manifest-dir output/manifests
                                            - name: Upload coverage
                                              if: always()
                                              uses: actions/upload-artifact@v4
                                              with:
                                                name: coreset-coverage
                                                path: experiments/3_coreset_engineering/htmlcov
                                      ```

                                      Troubleshooting
                                      ---------------

                                      - If tests fail due to missing AWS access, ensure secrets are populated and the test-only bucket is available.
                                      - Mismatched Python versions may cause subtle differences; ensure CI uses Python 3.12.
                                      - If dependency resolution fails, run `uv sync` locally to surface issues and ensure `uv.lock` is committed.
