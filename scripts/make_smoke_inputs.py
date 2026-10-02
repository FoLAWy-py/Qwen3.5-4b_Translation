"""Create complete near-budget samples solely to stress memory, not train quality."""
import copy

from transformers import AutoTokenizer

from witrans_tools.common import BASE_DIR, read_jsonl, write_jsonl
from witrans_tools.data import encode_example


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
    sample = next(r for r in read_jsonl("data/generated/pilot_reviewed.jsonl") if r["id"] == "pilot-academic-004")
    for budget in (2048, 1536, 1024):
        previous = None
        for repeats in range(1, 200):
            row = copy.deepcopy(sample)
            row["id"] = row["group_id"] = f"memory-smoke-{budget}"
            row["input"]["text"] = "\n".join([sample["input"]["text"]] * repeats)
            row["output"]["translation"] = "\n".join([sample["output"]["translation"]] * repeats)
            row["review"] = {"status": "pending", "notes": "Repetitive complete text for memory stress only"}
            try:
                encoded = encode_example(tokenizer, row, budget)
            except ValueError:
                break
            previous = row
        if previous is None:
            raise ValueError("无法构造预算内样本")
        length = len(encode_example(tokenizer, previous, budget)["input_ids"])
        write_jsonl(f"data/generated/memory_smoke_{budget}.jsonl", [previous])
        print(f"budget={budget}, actual={length}")


if __name__ == "__main__":
    main()
