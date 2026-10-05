
## Github basics

### Standard Workflow

- For every work session, begin by pulling latest changes from `main` if there's any: `git pull origin main`.
- Try to limit editing to files within your section to prevent merging issues.
- For every new feature/ set of changes, create a new local branch: `git checkout -b <your-branch-name>`
- To push code onto the repository:
  - `git add <files-to-be-added>` or `git add .` for all files.
  - `git commit -m <commit-message>`.
  - `git push origin <your-branch-name>`
  - Then follow instructions either in the terminal or on the repo's home page to open a Pull Request (PR).
  - (Optionally) Asks for review from another team member.
  - Finally, merge the PR into `main`.

> [!Note]
> - Only open a PR after you are sure that the codes are stable and will not break existing features.
> - Never (force) pushes directly to `main`.
