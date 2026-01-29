# Follow these rules when contributing to the repo
  
## How to create a pull request

1. **Start from an issue**
   - Before opening a PR, create or pick an existing GitHub issue that describes the problem or task.
   - Use the issue to capture scope, discussion, and acceptance criteria.

2. **Create a feature branch**
   - Always prefix the feature branch with Project Number based on your project number.
   - Branch from `main` using a descriptive name, e.g. `p19/feat/tokenizer-selection-improvements` or `p7/fix/eval-lite-metrics`.
   - Keep each PR focused on a single logical change whenever possible. Avoid adding too many commits/files to a single PR.
   - Short lived feature branches are easier and quicker to review and merge. Hence short lived feature branches (>1-2 days) are recommended.
   - Contributors are recommended to rebase there feature branches with main atleast twice everyday

3. **Implement and keep PRs small**
   - Prefer multiple small PRs over one very large one.
   - Avoid mixing refactors, new features, and formatting-only changes in the same PR.
   - Rebase your feature branch with main before raising the PR

4. **Link the PR to an issue**
   - In the PR description, reference the issue using GitHub keywords so it auto-closes when merged, e.g.:
     - `Closes #123` (preferred)
     - or `Fixes #123`, `Resolves #123`
   - If the PR is related but should **not** close the issue, use non-closing language like `Related to #123`.

5. **Describe what and why**
   - Start the PR description with a short summary of the change.
   - Add a brief “Why” section explaining the motivation or context.
   - List any trade-offs, known limitations, or follow-up work.
   - Limit to 50 characters: Keep the subject line short for readability in various Git tools.
   - Prefix: Use type prefixes like feat:, fix:, docs:, style:, refactor:, test:, chore:, perf: to categorize the commit.
   - Atomic Commits: Each commit should represent a single, logical change. - Avoid bundling unrelated changes (e.g., a bug, fix, and a refactor) into one commit.

6. **Add tests and docs**
   - Mention what tests you ran (and add them if missing).
   - Update `README`s or `docs/` where relevant and link to the updated files in the PR.

7. **Run checks before requesting review**
   - Run pre-commit hooks locally (formatting and linting).
   - Ensure notebooks/scripts do not use hard-coded, machine-specific paths.
   - Before committing, run the pre-commit hooks defined in `.pre-commit-config.yaml`:
     - Install once per machine: `pip install pre-commit && pre-commit install`
     - Optionally check everything: `pre-commit run --all-files`. This ensures formatting (Black, isort) and linting (Ruff) pass locally before you open a PR.
     - If there are any issues found while running pre-commit hooks locally vs running on CI then please execute the below set of commands in order to debug and run (also if pre-commit config has changed then one might have to clean the environment using below commands and then try git commit again) -
       - pre-commit clean
       - rm -rf ~/.cache/pre-commit
       - pre-commit install
       - pre-commit run --all-files

8. **Request and respond to reviews**
   - Add at least two reviewers.
   - Address review comments via follow-up commits; summarize major changes in a comment if the PR evolves significantly.

## Code and environment

- When working with files and paths (for example, in notebooks or scripts), avoid hard‑coded absolute paths that are specific to one machine or OS.
- Prefer repo‑relative paths and `pathlib.Path` so code runs unchanged on Windows, macOS, and Linux.

```bash
from pathlib import Path

file_path = Path("experiments/tokenizer/selection/ds_tokenizer.json")
```

## References

1. [Github Best Practices](https://dev.to/pwd9000/github-repository-best-practices-23ck)
2. [pre-commit](https://pre-commit.com/)
