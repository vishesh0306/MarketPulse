"""Reads a phase's run-summary JSON from logs/ and prints PASS/FAIL against the
mechanically checkable exit criteria for that phase (record-count reconciliation,
non-zero output, etc.). Judgment-only checks like "do the sentiment scores look sane"
require reading the actual output and are out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root isn't on sys.path when this is run directly (python scripts/validate_phase.py)
# rather than as a module, since Python only adds the script's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_settings  # noqa: E402

CheckResult = tuple[bool, str]


def find_latest_summary(log_dir: Path, phase: str) -> Path | None:
    """Returns the most recent run_<phase>_*.json in log_dir, or None if none exist."""
    candidates = sorted(log_dir.glob(f"run_{phase}_*.json"))
    return candidates[-1] if candidates else None


def _check_scraper(summary: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    total = summary.get("total_collected", 0)
    target = summary.get("min_tweets_target") or load_settings().scraper.min_tweets_target
    checks.append((total >= target, f"total_collected >= min_tweets_target ({total}/{target})"))

    per_hashtag = summary.get("per_hashtag", [])
    hashtags_with_data = [entry["hashtag"] for entry in per_hashtag if entry.get("collected", 0) > 0]
    checks.append(
        (len(hashtags_with_data) == len(per_hashtag), f"every requested hashtag collected >0 tweets ({hashtags_with_data})")
    )
    return checks


def _check_processing(summary: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    in_count = summary.get("in", 0)
    out_count = summary.get("out", 0)
    rejected = summary.get("rejected", 0)
    deduped = summary.get("deduped", 0)
    checks.append(
        (in_count == out_count + rejected + deduped, f"in ({in_count}) == out + rejected + deduped ({out_count + rejected + deduped})")
    )
    checks.append((out_count > 0, f"out > 0 (got {out_count})"))
    return checks


def _check_signals(summary: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    buckets = summary.get("buckets", 0)
    checks.append((buckets > 0, f"buckets > 0 (got {buckets})"))
    checks.append((len(summary.get("hashtags", [])) > 0, "at least one hashtag present in output"))
    return checks


_PHASE_CHECKS = {
    "scraper": _check_scraper,
    "processing": _check_processing,
    "signals": _check_signals,
}


def check_phase(phase: str, log_dir: Path = Path("logs")) -> int:
    """Prints PASS/FAIL checks for the given phase number/name. Returns process exit code."""
    summary_path = find_latest_summary(log_dir, phase)
    if summary_path is None:
        print(f"[FAIL] No run-summary JSON found for phase '{phase}' in {log_dir}/")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Loaded {summary_path}")

    check_fn = _PHASE_CHECKS.get(phase)
    if check_fn is None:
        print(f"[INFO] No mechanical checks defined for phase '{phase}' — known phases: {sorted(_PHASE_CHECKS)}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    results = check_fn(summary)
    all_passed = True
    for passed, description in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")
        all_passed = all_passed and passed

    return 0 if all_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a pipeline phase against its run-summary log.")
    parser.add_argument("--phase", type=str, required=True, help="Phase name: 'scraper', 'processing', or 'signals'.")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()
    sys.exit(check_phase(args.phase, Path(args.log_dir)))


if __name__ == "__main__":
    main()
