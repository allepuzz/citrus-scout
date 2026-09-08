## What changes

<!-- One or two sentences. What this PR does, not how. -->

## Why

<!-- The problem it solves. Link the issue if there is one: Closes #N -->

## How to test

<!-- Concrete steps to verify the change. Exact command where applicable. -->

```bash
uv run pytest
```

## Impact on results

<!-- Only if this touches models, data or metrics. Delete this section otherwise.
     Include before/after metrics: PR-AUC, F1, PPV. Never accuracy alone. -->

| Metric | Before | After |
|---|---|---|
| PR-AUC | | |
| F1 | | |
| PPV @ 2% prevalence | | |

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run pytest` passes
- [ ] No credentials, data or checkpoints in the diff
- [ ] Config changes are documented
- [ ] README updated if usage changed
