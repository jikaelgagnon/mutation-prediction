"""Apply and stage Ruff fixes without blocking on remaining diagnostics."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    files = sys.argv[1:]
    if not files:
        return

    # pre-commit temporarily hides unstaged edits before invoking this hook.
    original = {filename: Path(filename).read_bytes() for filename in files}
    subprocess.run(
        ["ruff", "check", "--fix", "--no-unsafe-fixes", "--exit-zero", "--", *files],
        check=False,
    )
    subprocess.run(["ruff", "format", "--", *files], check=False)

    changed = [
        filename
        for filename in files
        if Path(filename).read_bytes() != original[filename]
    ]
    if changed:
        # Stage only files Ruff changed, so pre-commit can continue immediately.
        subprocess.run(["git", "add", "--", *changed], check=True)


if __name__ == "__main__":
    main()
