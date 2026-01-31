# Team 9 Git Workflow Guide

## Code Organization

Create and maintain this structure in our team folder:

```
experiments/9_training_stack_optimisation_and_cost_governor/
├── training/               # Core training pipeline
│   ├── framework/         # DeepSpeed
│   ├── data_pipeline/     # Data loading
│   └── checkpoints/       # Checkpoint management
├── monitoring/            # Metrics & profiling
│   ├── gpu_monitor/
│   ├── throughput/
│   └── cost_tracking/
├── halt_system/           # Automatic HALT
│   ├── triggers/
│   ├── notifications/
│   └── recovery/
├── configs/               # Per-stage configs
│   ├── stage1_1b_dense/
│   ├── stage2_3b_moe/
│   ├── stage3_8b_dense/
│   └── stage4_70b_moe/
├── scripts/               # Automation
└── docs/                  # Documentation
```

---

## Conflict Prevention Strategies

### 1. Integration Branch for Complex Features

**When to use:** Features requiring multiple people (e.g., HALT system with triggers + notifications + recovery)

**Naming convention:**
```bash
# Main integration branch for main task
p09/feat/<feature-name>-integration

# Sub-branches for individual components
p09/feat/<feature-name>-<component>
```

**Example workflow:**

```bash
# One can create integration branch
git checkout main
git pull origin main
git checkout -b p09/feat/halt-system-integration
git push -u origin p09/feat/halt-system-integration

# Members can create sub-branches from integration branch
git checkout p09/feat/halt-system-integration
git pull origin p09/feat/halt-system-integration
git checkout -b p09/feat/halt-system-triggers
# ... work on triggers ...
git push -u origin p09/feat/halt-system-triggers

# Open PR: p09/feat/halt-system-triggers → p09/feat/halt-system-integration
# Once all sub-branches merged to integration branch and integration tested
# Open PR: p09/feat/halt-system-integration → main
```

**Real examples:**
```
Integration branches:
├── p09/feat/halt-system-integration
│   ├── p09/feat/halt-system-triggers
│   ├── p09/feat/halt-system-notifications
│   └── p09/feat/halt-system-recovery
│
├── p09/feat/cost-tracking-integration
│   ├── p09/feat/cost-tracking-metrics
│   ├── p09/feat/cost-tracking-reporting
│   └── p09/feat/cost-tracking-budget-enforcement
│
└── p09/feat/checkpoint-integration
    ├── p09/feat/checkpoint-save
    ├── p09/feat/checkpoint-restore
    └── p09/feat/checkpoint-s3-upload
```

### 2. Pre-commit Hooks

**Setup (run once per machine):**
```bash
cd <repository-root>
pip install pre-commit
pre-commit install
```

**Before every commit:**
```bash
# Automatically runs on git commit
# Or manually check all files:
pre-commit run --all-files
```

**What it checks:**
- Code formatting (Black, isort)
- Linting (Ruff)
- No hardcoded paths
- No committed secrets
- YAML validation

---

## Daily Workflow

### ⚠️ IMPORTANT: What to Rebase With?

**Choose based on your branch type:**

| Your Branch Type | Rebase With | Example |
|-----------------|-------------|---------|
| Sub-branch (targets integration) | Integration branch | `p09/feat/halt-system-triggers` → rebase with `p09/feat/halt-system-integration` |
| Standalone feature (targets main) | main | `p09/feat/gpu-monitoring` → rebase with `main` |
| Integration branch itself | main | `p09/feat/halt-system-integration` → rebase with `main` |

### Morning (Start of Day)

**Option A: For standalone features (targeting main)**
```bash
# 1. Update your local main
git checkout main
git pull origin main

# 2. Rebase your feature branch with main
git checkout p09/feat/your-feature
git rebase main

# 3. If conflicts occur, resolve them:
#    - Fix conflicts in files
#    - git add <resolved-files>
#    - git rebase --continue

# 4. Force push (rebase rewrites history)
git push -f origin p09/feat/your-feature
```

**Option B: For sub-branches (targeting integration branch)**
```bash
# 1. Update the integration branch
git checkout p09/feat/halt-system-integration
git pull origin p09/feat/halt-system-integration

# 2. Rebase your sub-branch with integration branch
git checkout p09/feat/halt-system-triggers
git rebase p09/feat/halt-system-integration

# 3. Resolve conflicts if any, then force push
git push -f origin p09/feat/halt-system-triggers
```

> **Note:** After a rebase, you must use `git push -f` (force push) if your branch was already pushed to the remote, because rebase rewrites commit history. For new branches, a normal push is sufficient.

### During Work (Commit Often)

```bash
# Make changes, then:
git add <changed-files>
git commit -m "feat(component): short description

- Detailed change 1
- Detailed change 2

Related to #issue-number"

# Pre-commit hooks run automatically
# Fix any issues they report, then commit again
```

### Evening (Before EOD - Second Daily Rebase)

**Option A: For standalone features (targeting main)**
```bash
# 1. Update main again
git checkout main
git pull origin main

# 2. Rebase your feature branch (second time today)
git checkout p09/feat/your-feature
git rebase main

# 3. Resolve conflicts if any, then force push
git push -f origin p09/feat/your-feature

# 4. Post update in Telegram
#    "EOD: Working on p09/feat/gpu-monitoring, files: monitoring/gpu_monitor.py"
```

**Option B: For sub-branches (targeting integration branch)**
```bash
# 1. Update the integration branch
git checkout p09/feat/halt-system-integration
**For standalone features (PR to main):**
```bash
# 1. Ensure you're up to date with main
git checkout main
git pull origin main
git checkout p09/feat/your-feature
git rebase main
git push -f origin p09/feat/your-feature

# 2. Run pre-commit one final time
pre-commit run --all-files

# 3. Open PR on GitHub
#    - Base: main
#    - Compare: p09/feat/your-feature
#    - Link issue with "Closes #123"
#    - Request review from at least 1 team member
```

**For sub-branches (PR to integration branch):**
```bash
# 1. Ensure you're up to date with integration branch
git checkout p09/feat/halt-system-integration
git pull origin p09/feat/halt-system-integration
git checkout p09/feat/halt-system-triggers
git rebase p09/feat/halt-system-integration
git push -f origin p09/feat/halt-system-triggers

# 2. Run pre-commit one final time
pre-commit run --all-files

# 3. Open PR on GitHub
#    - Base: p09/feat/halt-system-integration
#    - Compare: p09/feat/halt-system-triggers
#    - Link issue with "Related to #123"
#    - Request review from integration branch own
# 1. Ensure you're up to date
git checkout main
git pull origin main
git checkout p09/feat/your-feature
git rebase main
git push -f origin p09/feat/your-feature

# 2. Run pre-commit one final time
pre-commit run --all-files

# 3. Open PR on GitHub
#    - Base: main (or integration branch if applicable)
#    - Compare: p09/feat/your-feature
#    - Link issue with "Closes #123"
#    - Request review from at least 1 team member
```

### After PR is Merged

```bash
# 1. Update main
git checkout main
git pull origin main

# 2. Delete local branch
git branch -d p09/feat/your-feature

# 3. Delete remote branch
git push origin --delete p09/feat/your-feature

# 4. Start next feature from clean main
git checkout -b p09/feat/next-feature
```

---

## Quick Commands Reference

```bash
# Check which branch you're on
git branch

# See what changed
git status
git diff

# Abandon changes and start over
git checkout -- <file>

# Abort a problematic rebase
git rebase --abort

# See recent commits
git log --oneline -10

# Update someone else's branch (if helping)
git fetch origin
git checkout p09/feat/their-branch
git pull origin p09/feat/their-branch
```

---

If blocked or confused ask in Telegram

Keep branches **short-lived** (1-2 days max)

Run **pre-commit** before pushing
