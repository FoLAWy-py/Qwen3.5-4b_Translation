"""Freeze a 100-record stratified screen; keep the full 1,000-record test untouched."""
from collections import defaultdict
from pathlib import Path

from witrans_tools.common import fingerprint, read_jsonl, write_json, write_jsonl


def main():
    destination = Path("data/prepared/v1")
    report = {}
    for split in ("dev", "test"):
        output = destination / f"{split}_screen.jsonl"
        if output.exists():
            raise ValueError("筛查集合已冻结，不能覆盖")
        rows = read_jsonl(destination / f"{split}.jsonl")
        groups = defaultdict(list)
        for row in rows:
            variant = int(row["id"].split("-")[-3] if row["id"].endswith("zh-CN") else row["id"].split("-")[-2])
            if variant in (5, 13):
                groups[row["group_id"]].append(row)
        screen = [r for group in sorted(groups) for r in sorted(groups[group], key=lambda r: r["id"])]
        if len(screen) != 100 or any(len(g) != 4 for g in groups.values()):
            raise ValueError("筛查集合数量错误")
        write_jsonl(output, screen)
        report[split] = {"count": len(screen), "family_count": len(groups), "sha256": fingerprint(screen)}
    write_json(destination / "screen_manifest.json", report)
    print(report)


if __name__ == "__main__":
    main()
