"""Bind a human-written TSV decision to its exact inspected generated text."""
import argparse
from pathlib import Path
from witrans_tools.common import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', required=True)
    parser.add_argument('--tsv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists(), 'Preserve explicit reading batch'
    rows = {r['id']: r for r in read_jsonl(args.stem + '.jsonl')}
    readings = []
    for line in Path(args.tsv).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        rid, verdict, note, translation = line.split('|', 3)
        row = rows[rid]
        assert translation == row['prediction']['translation'], rid
        assert verdict in ('pass', 'minor', 'major', 'critical') and note
        readings.append(dict(id=rid, input=row['input'], translation=translation,
                             verdict=verdict, note=note))
    assert len(readings) == len({r['id'] for r in readings})
    write_jsonl(args.output, readings)
    print({'explicit_decisions': len(readings), 'output': args.output})


if __name__ == '__main__':
    main()
