# Git Remote Policy

When committing, pushing, creating branches, or creating PRs, ALWAYS use the `origin` remote (billy-enrizky/openbrowser-ai).

Do NOT use the `csc490` remote (UofT-CSC490-W2026/OpenBrowser-AI) unless the user explicitly requests it.

## Commands

```bash
# Push
git push origin <branch>

# Create PR
gh pr create --repo billy-enrizky/openbrowser-ai ...

# Set upstream
git push -u origin <branch>
```
