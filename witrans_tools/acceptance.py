"""Numeric quality gate over completed reviewer decisions, never an auto-grader."""
from collections import Counter

CATEGORIES = ("daily", "travel", "food", "academic", "hard")
DIRECTIONS = ("en", "zh-CN")

def quality_gate(decisions):
    rows = list(decisions)
    failures = []
    if len(rows) < 600:
        failures.append("fewer_than_600_rows")
    if len({r['group_id'] for r in rows}) < 300:
        failures.append("fewer_than_300_source_groups")
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate reviewed IDs")
    for row in rows:
        if row['verdict'] not in ("pass", "minor", "major", "critical"):
            raise ValueError("Unreviewed verdict")
        if row['category'] not in CATEGORIES or row['target_lang'] not in DIRECTIONS:
            raise ValueError("Unknown evaluation stratum")
        if any(type(row[k]) is not bool for k in ("format_valid", "ended", "language_correct")):
            raise ValueError("Explicit reviewer/format flags required")
    n = len(rows)
    counts = Counter(r['verdict'] for r in rows)
    if not n or counts['pass'] / n < .95:
        failures.append("pass_below_95_percent")
    if n and (counts['major'] + counts['critical']) / n > .01:
        failures.append("major_above_1_percent")
    if counts['critical']:
        failures.append("critical_error_present")
    if any(not all(r[k] for k in ("format_valid", "ended", "language_correct")) for r in rows):
        failures.append("format_eos_or_direction_failure")
    for category in CATEGORIES:
        for target in DIRECTIONS:
            bucket = [r for r in rows if r['category'] == category and r['target_lang'] == target]
            if len(bucket) < 60:
                failures.append(f"coverage:{category}/{target}")
            if bucket and (sum(r['verdict'] == 'pass' for r in bucket) / len(bucket) < .90 or sum(r['verdict'] in ('major', 'critical') for r in bucket) / len(bucket) > .03):
                failures.append(f"quality:{category}/{target}")
    return {"semantic_numeric_gate": not failures, "failures": failures, "rows": n, "verdicts": dict(counts),
        "release_approved": False, "remaining_checks": "Source audit, context/paragraph coverage, paired baseline/regression, confidence intervals and performance are separate mandatory checks"}
