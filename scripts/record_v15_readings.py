"""Export explicit Codex source readings, bound to the exact displayed translation."""
import argparse
import json
from pathlib import Path
from witrans_tools.common import fingerprint, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', required=True)
    parser.add_argument('--readings', required=True)
    args = parser.parse_args()
    outputs = {r['id']: r for r in read_jsonl(args.stem + '.jsonl')}
    path = Path(args.stem + '-manual.jsonl')
    previous = read_jsonl(path) if path.exists() else []
    decisions = {r['id']: r for r in previous}
    assert len(decisions) == len(previous)
    for reading in read_jsonl(args.readings):
        row = outputs[reading['id']]
        assert row['input'] == reading['input'], 'Displayed input changed'
        assert row['prediction']['translation'] == reading['translation'], 'Displayed output changed'
        assert reading['verdict'] in ('pass', 'minor', 'major', 'critical') and reading['note']
        decision = dict(id=row['id'], reviewer='Codex', generation_hash=fingerprint(row),
                        verdict=reading['verdict'], note=reading['note'], language_correct=True)
        assert row['id'] not in decisions or decisions[row['id']] == decision, 'Preserve prior reading'
        decisions[row['id']] = decision
    write_jsonl(path, list(decisions.values()))
    print({'explicit_readings': len(decisions), 'stem': args.stem})


if __name__ == '__main__':
    main()
