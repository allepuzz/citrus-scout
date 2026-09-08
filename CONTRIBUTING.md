# Development guide

## Workflow

`main` is protected: never commit to it directly. Every change lands through a Pull Request.

```bash
# 1. Always branch from an up-to-date main
git checkout main && git pull

# 2. New branch
git checkout -b feat/short-description

# 3. Work and commit
git add -p                  # review what you stage, hunk by hunk
git commit -m "feat: add leaf dataset loader"

# 4. Push and open a PR
git push -u origin feat/short-description
gh pr create --fill
```

### Branch naming

| Prefix | Use |
|---|---|
| `feat/` | New functionality |
| `fix/` | Bug fix |
| `refactor/` | Internal change, no behaviour change |
| `exp/` | Model or data experiment |
| `docs/` | Documentation only |
| `chore/` | Tooling, dependencies, CI |

## Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <imperative, lowercase description>

[optional body explaining why]
```

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `exp`.

```bash
# Good
feat: add configurable PPV-at-prevalence metric
fix: prevent train/val leakage when stratifying

# Bad
changes
update
fixed the bug
```

Write the **why** in the body, not the what — the what is already in the diff.

## Before opening a PR

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

The pre-commit hooks do this automatically. Install them once:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Hard rules

**Never commit:**
- Credentials (`kaggle.json`, `.env`, tokens) — `.gitignore` and the hooks block these,
  but review your diff anyway
- Data or imagery — managed with DVC
- Model checkpoints — these go to W&B or separate storage
- Notebook outputs — `nbstripout` strips them automatically

If a secret ever reaches `main`, **deleting it in a later commit is not enough**: it stays in
history. Rotate the credential immediately and rewrite history.

## Experiments

Experiments live on `exp/` branches. An experiment counts as reproducible when:

1. Its config is in `configs/` and version-controlled
2. The seed is fixed and recorded
3. The data version is pinned with DVC
4. Metrics are logged to W&B

A PR that changes the model must report **PR-AUC, F1 and PPV at real prevalence**.
Accuracy alone is not accepted as evidence: at 2% prevalence, a model that always predicts
"healthy" scores 98% accuracy and is worthless.

## Code layout

- `src/citrus_scout/` is an installable package. Imports are absolute:
  `from citrus_scout.data import LeafDataset`
- Logic belongs in the package, **not in notebooks**. Notebooks explore and visualise, but
  any reusable code moves into `src/`.
- Training is always launched through the CLI with a config file, never with hardcoded
  constants in a script. That is what lets the same code run locally and on Colab.
