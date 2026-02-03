# Rebase Feature Branches onto `staging`

## Purpose

This repository uses **`staging` as the integration branch**.

All development must follow:

```text
<pxx>/feat/* → staging → main
```

### Mandatory Rules

- Create feature branches from `staging`
- Open PRs into `staging`
- Do NOT branch from `main`
- Do NOT open PRs to `main`
- Always use **Squash & Merge**

---

## When should you use this guide?

Use this if:

- You already created a feature branch from `main`
- Your feature branch is ahead/behind main
- Your PR targets `main`
- You need to sync with latest staging

---

## One-Time Migration of Existing Feature Branch

These steps work for:

- ahead of main
- behind main
- both

Same commands for all cases.

---

### Step 1 — Checkout your feature branch

```git
git checkout <feature-branch-name>
```

---

### Step 2 — Ensure clean working tree

```git
git status
```

#### If clean → continue

#### If NOT clean

Option A (recommended):

```git
git add .
git commit -m "WIP: before staging rebase"
```

Option B:

```git
git stash
```

You need to run 'git stash pop' later only if you ran 'git stash'

---

### Step 3 — Fetch latest remote state (SAFE)

```git
git fetch origin
```

This only updates remote tracking references.  
It does NOT modify your code.

---

### Step 4 — Rebase onto staging

```git
git rebase origin/staging
```

---

### Step 5 — Resolve conflicts (if any)

```git
git add .
git rebase --continue
```

Abort if needed:

```git
git rebase --abort
```

If rebase gets confusing then JUST RESTART REBASE AGAIN. NEVER FIGHT a messy rebase.

---

### Step 6 — Restore stash (only if used)

```git
git stash pop
```

---

### Step 7 — Push branch

First time:

```git
git push -u origin <feature-branch>
```

After future rebases:

```git
git push --force-with-lease
```

---

## Validate Rebase

### Check graph

```git
git log --oneline --graph --decorate -15
```

Expected:

```git
staging → your commits
```

---

## Update PR Base

In GitHub:

1. Open PR
2. Click Edit
3. Change base: main → staging
4. Save

Ignore warnings about outdated comments.

---

## Daily Workflow (Recommended)

```git
git checkout <feature-branch>
git fetch origin
git rebase origin/staging
git push --force-with-lease
```

---

## Creating New Feature Branches (Going Forward)

Always start from staging:

```git
git checkout staging
git pull
git checkout -b feature/<name>
```

Open PR:

```git
feature → staging
```

Never:

```text
feature → main
```

---

## Merge Strategy

Use only:

- Squash & Merge

Do NOT use:

- Rebase & Merge
- Merge commits

---

## Common Errors

### Unstaged changes

```text
cannot rebase: You have unstaged changes
```

Fix:

```git
git stash
```

### No upstream branch

```text
fatal: no upstream branch
```

Fix:

```git
git push -u origin <feature-branch>
```

---

## Cheatsheet

Rebase:

```git
git checkout <branch>
git fetch origin
git rebase origin/staging
git push --force-with-lease
```

New branch:

```git
git checkout staging
git pull
git checkout -b feature/<name>
```

---

## Final Summary

- branch from staging
- PR to staging
- fetch before rebase
- rebase often
- squash merge only
- never push to main
