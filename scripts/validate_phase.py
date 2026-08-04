"""Reads a phase's run-summary JSON from logs/ and prints PASS/FAIL against the mechanically
checkable items in VALIDATION_CHECKLIST.md. Judgment-only items are left for human/agent review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_latest_summary(log_dir: Path, phase: str) -> Path | None:
    """Returns the most recent run_<phase>_*.json in log_dir, or None if none exist."""
    candidates = sorted(log_dir.glob(f"run_{phase}_*.json"))
    return candidates[-1] if candidates else None


def check_phase(phase: str, log_dir: Path = Path("logs")) -> int:
    """Prints PASS/FAIL checks for the given phase number/name. Returns process exit code."""
    summary_path = find_latest_summary(log_dir, phase)
    if summary_path is None:
        print(f"[FAIL] No run-summary JSON found for phase '{phase}' in {log_dir}/")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Loaded {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Phase-specific mechanical checks are added as each phase is implemented.
    print(f"[INFO] Phase '{phase}' summary loaded; mechanical checks not yet implemented for this phase.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a pipeline phase against its run-summary log.")
    parser.add_argument("--phase", type=str, required=True, help="Phase name, e.g. '1', '2', 'scraper'.")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()
    sys.exit(check_phase(args.phase, Path(args.log_dir)))


if __name__ == "__main__":
    main()
