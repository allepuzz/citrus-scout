"""Block direct commits to protected branches.

Server-side rulesets require GitHub Pro on private repositories, so this hook
enforces the same rule locally instead. It is a guard rail, not a security
control: anyone can bypass it with `--no-verify`.

Wired up as a pre-commit hook; see `.pre-commit-config.yaml`.
"""

from __future__ import annotations

import subprocess
import sys

PROTECTED_BRANCHES = frozenset({"main", "master"})


def current_branch() -> str | None:
    """Return the checked-out branch, or None when detached or not in a repo."""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        # Detached HEAD (rebase, bisect, checkout of a tag) — nothing to protect.
        return None
    return result.stdout.strip()


def main() -> int:
    branch = current_branch()
    if branch is None or branch not in PROTECTED_BRANCHES:
        return 0

    print(
        f"\n  Refusing to commit directly to '{branch}'.\n\n"
        "  Move your work onto a branch and open a PR:\n\n"
        "      git checkout -b feat/your-change\n"
        "      git push -u origin feat/your-change\n"
        "      gh pr create --fill\n\n"
        "  If the commit is already staged, the branch will carry it across.\n"
        "  Override only when you know why:  git commit --no-verify\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
