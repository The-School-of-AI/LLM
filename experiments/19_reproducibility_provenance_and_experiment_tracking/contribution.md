Follow these rules when contributing to the repo.

## Pull requests and review

1. The main branch is protected. Create a pull request in your assigned folder. Treat these folders as temporary spaces for early experiments. If you plan to share code across folders or change the structure, discuss it with the team first.

2. Each pull request needs two reviewers. Reviewers should tag themselves to avoid duplicate reviews. Write a clear pull request description. Add a README with steps to run and reproduce the results. Include screenshots or result visuals when they help others validate the work.

3. Reviewers can also add screenshots during review. This helps others understand the results and approve faster when full testing is not required.

### How to create a pull request

1. **Start from an issue**
   - Before opening a PR, create or pick an existing GitHub issue that describes the problem or task.
   - Use the issue to capture scope, discussion, and acceptance criteria.

2. **Create a feature branch**
   - Branch from `main` using a descriptive name, e.g. `feature/tokenizer-selection-improvements` or `fix/eval-lite-metrics`.
   - Keep each PR focused on a single logical change whenever possible.

3. **Implement and keep PRs small**
   - Prefer multiple small PRs over one very large one.
   - Avoid mixing refactors, new features, and formatting-only changes in the same PR.

4. **Link the PR to an issue**
   - In the PR description, reference the issue using GitHub keywords so it auto-closes when merged, e.g.:
     - `Closes #123` (preferred)
     - or `Fixes #123`, `Resolves #123`
   - If the PR is related but should **not** close the issue, use non-closing language like `Related to #123`.

5. **Describe what and why**
   - Start the PR description with a short summary of the change.
   - Add a brief “Why” section explaining the motivation or context.
   - List any trade-offs, known limitations, or follow-up work.

6. **Add tests and docs**
   - Mention what tests you ran (and add them if missing).
   - Update `README`s or `docs/` where relevant and link to the updated files in the PR.

7. **Run checks before requesting review**
   - Run pre-commit hooks locally (formatting and linting).
   - Ensure notebooks/scripts do not use hard-coded, machine-specific paths.

8. **Request and respond to reviews**
   - Add at least two reviewers.
   - Address review comments via follow-up commits; summarize major changes in a comment if the PR evolves significantly.

## Code and environment

4. When working with files and paths (for example, in notebooks or scripts), avoid hard-coded absolute paths that are specific to one machine or OS. Prefer repo-relative paths and `pathlib.Path` so code runs unchanged on Windows, macOS, and Linux.
```bash
from pathlib import Path

file_path = Path("experiments/tokenizer/selection/ds_tokenizer.json")
# this work unchanged on Windows, Linux, and macOS because Path(...) will normalize separators for the current OS.
```

## References
1. [Github Best Practices](https://dev.to/pwd9000/github-repository-best-practices-23ck)