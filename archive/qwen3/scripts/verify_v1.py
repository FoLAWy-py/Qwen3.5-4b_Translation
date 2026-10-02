"""Check every finalized reference's length, loss encoding, and constraints."""
from transformers import AutoTokenizer

from witrans_tools.common import BASE_DIR, read_jsonl, write_json
from witrans_tools.data import check_constraints, encode_example, validate_record


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
    rows = read_jsonl("data/v1_references.jsonl")
    lengths = []
    for row in rows:
        validate_record(row, True, True)
        if check_constraints(row["input"]["text"], row["output"]["translation"]):
            raise ValueError(f"保留项不符: {row['id']}")
        lengths.append(len(encode_example(tokenizer, row, 1024)["input_ids"]))
    result = {"checked": len(rows), "max_tokens": max(lengths), "min_tokens": min(lengths),
              "constraint_failures": 0, "overlength": 0,
              "normal_input_count": sum(not r["input"]["context"] and not r["input"]["glossary"] for r in rows)}
    write_json("runs/v1-reference-checks.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
