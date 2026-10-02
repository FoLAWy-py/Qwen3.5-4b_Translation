"""Preserve the frozen originals and correct two reversed bilingual sources."""
from witrans_tools.common import read_jsonl, write_jsonl, write_json, now, fingerprint
from scripts.build_v2 import reviewed


def main():
    rows = read_jsonl("data/v2_train_references.jsonl")
    corrected_inputs, changes = [], []
    for row in rows:
        if row["id"].split("-")[2] in ("115", "117"):
            old = fingerprint(row)
            row["input"]["text"], row["output"]["translation"] = row["output"]["translation"], row["input"]["text"]
            row["id"] += "-sourcefix"
            reviewed(row, "Corrected reversed bilingual authoring fields before training; original preserved")
            changes.append({"id": row["id"], "original_hash": old, "corrected_hash": fingerprint(row)})
            corrected_inputs.append({k: v for k, v in row.items() if k not in ("review", "output")})
    write_jsonl("data/v2_train_references_corrected.jsonl", rows)
    write_jsonl("data/v2_sourcefix_inputs.jsonl", corrected_inputs)
    dev = read_jsonl("data/prepared/v2/dev.jsonl")
    for row in dev:
        if row["id"].startswith("v2-dev-020-"):
            old = fingerprint(row)
            row["input"]["text"], row["output"]["translation"] = row["output"]["translation"], row["input"]["text"]
            row["id"] += "-sourcefix"
            reviewed(row, "Corrected reversed bilingual fields before the final optimization run")
            changes.append({"id": row["id"], "original_hash": old, "corrected_hash": fingerprint(row)})
    write_jsonl("data/prepared/v2/dev_corrected.jsonl", dev)
    write_jsonl("data/prepared/v2/dev_screen_corrected.jsonl", dev)
    write_json("runs/v2-source-corrections.json", {"at": now(), "reason": "Three authored bilingual pairs had reversed language fields; final run uses corrected train and dev, originals preserved", "changes": changes})


if __name__ == "__main__":
    main()
