Follow these rules when contributing to the repo.

## Pull requests and review

1. The main branch is protected. Create a pull request in your assigned folder. Treat these folders as temporary spaces for early experiments. If you plan to share code across folders or change the structure, discuss it with the team first.

2. Each pull request needs two reviewers. Reviewers should tag themselves to avoid duplicate reviews. Write a clear pull request description. Add a README with steps to run and reproduce the results. Include screenshots or result visuals when they help others validate the work.

3. Reviewers can also add screenshots during review. This helps others understand the results and approve faster when full testing is not required.



## Code and environment

4. When working with files and paths (for example, in notebooks or scripts), avoid hard-coded absolute paths that are specific to one machine or OS. Prefer repo-relative paths and `pathlib.Path` so code runs unchanged on Windows, macOS, and Linux.

## References
1. [Github Best Practices](https://dev.to/pwd9000/github-repository-best-practices-23ck)