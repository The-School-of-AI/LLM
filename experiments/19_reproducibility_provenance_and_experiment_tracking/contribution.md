# Follow these rules when contributing to the repo

- The main branch is protected. Create a pull request in your assigned folder. Treat these folders as temporary spaces for early experiments. If you plan to share code across folders or change the structure, discuss it with the team first.

- For easy and quick PR reviews, please maintain Atomic commits and rebase frequently with main to avoid diverging. 

- Short lived feature branches are easier and quicker to review and merge. Hence short lived feature branches (>1-2 days) are recommended. Avoid adding too many commits/files to a single PR.

- Recommended for contributors to rebase with main everyday 2-3 times. Before raising the PR rebase with main, check merge issues fix them on local and push the changes again.

- Each pull request needs two reviewers. Reviewers should tag themselves to avoid duplicate reviews. Write a clear pull request description. Add a README with steps to run and reproduce the results. Include screenshots or result visuals when they help others validate the work.

- Reviewers can also add screenshots during review. This helps others understand the results and approve faster when full testing is not required.

- Branch names should be descriptive, use prefixes to indicate their purpose, and use lowercase characters with hyphens for separation -
  - Prefixes: Categorize branches using a type prefix followed by a slash (e.g., feat/, bugfix/, hotfix/, docs/, release/). Prefix this with the project #. For example, if the Coresets Engineering team, which is Project 3, is creating the new feature then name it - <p3/feat/coresets-run.py>
  - Descriptive names: The name should concisely explain the purpose of the branch (e.g., feat/user-auth, not feat/login).
  - Use hyphens: Separate words with hyphens for readability (e.g., px/bugfix/fix-login-issue, not bugfix/fixLoginIssue or bugfix/fix_login_issue
  - Include issue numbers: If using a project management tool (like Jira or GitHub Issues), include the ticket number for easy tracking (e.g., px/feat/T-123-add-billing-module).
  
- Commit messages should be clear, concise, and informative, explaining both the change made and its rationale.
  - Subject Line:
    - Limit to 50 characters: Keep the subject line short for readability in various Git tools.
    - Prefix: Use type prefixes like feat:, fix:, docs:, style:, refactor:, test:, chore:, perf: to categorize the commit.
  - Body (optional):
    - Wrap at 72 characters: Ensure the body text is wrapped to 72 characters for readability.
    - Explain "what" and "why": Detail the motivation for the change and how it differs from previous behavior, not how it was implemented.
    - Atomic Commits: Each commit should represent a single, logical change. - Avoid bundling unrelated changes (e.g., a bug fix and a refactor) into one commit.

## References

- [Github Best Practices](https://dev.to/pwd9000/github-repository-best-practices-23ck)