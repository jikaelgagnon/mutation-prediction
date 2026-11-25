"""Run the split regression against the old and fixed code on the cluster.

Uses temporary source trees; never checks out files over the working tree.
The red phase must fail the membership assertion, not imports or inference.
"""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BASELINE = "fbb33032f53267c749c74f61b054c36a28acb9e8"


def run_tests(root, source, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE, help="Commit before the fix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="epsilon-splits-red-green-") as directory:
        temporary = Path(directory)
        archive = temporary / "baseline.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive}",
                args.baseline,
                "src",
            ],
            cwd=root,
            check=True,
        )
        baseline = temporary / "baseline"
        baseline.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(baseline, filter="data")
        print(f"RED: published membership test against {args.baseline}", flush=True)
        red = run_tests(
            root,
            baseline / "src",
            "-p",
            "test_prepare_published_splits.py",
            "-k",
            "test_default_export_keeps_exact_published_mutation_memberships",
            "-v",
        )
        output = red.stdout + red.stderr
        if not (
            red.returncode == 1
            and "PUBLISHED_SPLIT_MEMBERSHIP" in output
            and "Ran 1 test" in output
            and "FAILED (failures=1)" in output
        ):
            raise SystemExit(
                "Red phase did not produce the expected membership failure."
            )
        print("RED confirmed: old default exports the wrong mutations.", flush=True)

    print("GREEN: same test against the working source", flush=True)
    green = run_tests(
        root, root / "src", "-p", "test_prepare_published_splits.py", "-v"
    )
    if green.returncode:
        raise SystemExit(green.returncode)
    print("GREEN: existing target-preparation regression tests", flush=True)
    regression = run_tests(
        root, root / "src", "-p", "test_precompute_megascale_ddg.py", "-v"
    )
    if regression.returncode:
        raise SystemExit(regression.returncode)
    print("Red/green checks passed. Full data preparation can now run separately.")


if __name__ == "__main__":
    main()
