"""Print source/translation for Codex's individual semantic reading."""
import argparse
from witrans_tools.common import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--path", default="runs/v4-mining-start.jsonl")
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    for r in read_jsonl(args.path)[args.start:args.end]:
        print(r["id"], "|", r["input"]["text"], "| context:", r["input"]["context"],
            "| glossary:", r["input"].get('glossary', {}), "|", r.get("prediction", r.get("output", r.get("raw"))))

if __name__ == "__main__":
    main()
