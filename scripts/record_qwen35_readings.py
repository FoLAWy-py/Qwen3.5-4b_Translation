"""Record explicit inspected packet decisions without duplicating long translations."""
import argparse
import json
from pathlib import Path
from witrans_tools.common import fingerprint, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', required=True)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--decisions', required=True)
    args = parser.parse_args()
    packets = {p['id']: p for p in read_jsonl(args.packet)}
    rows = {r['id']: r for r in read_jsonl(args.stem + '.jsonl')}
    decisions = json.loads(Path(args.decisions).read_text(encoding='utf-8-sig'))
    path = Path(args.stem + '-manual.jsonl')
    old = read_jsonl(path) if path.exists() else []
    notes = {r['id']: r for r in old}
    assert len(notes) == len(old)
    assert set(decisions) <= set(packets)
    for rid, explicit in decisions.items():
        packet, row = packets[rid], rows[rid]
        assert packet['generation_hash'] == fingerprint(row), 'Inspected packet changed'
        assert packet['input'] == row['input'] and packet['raw'] == row.get('raw')
        assert packet['translation'] == row.get('prediction', {}).get('translation')
        verdict, note, language_correct = explicit
        assert verdict in ('pass', 'minor', 'major', 'critical')
        assert isinstance(note, str) and note.strip() and type(language_correct) is bool
        decision = dict(id=rid, reviewer='Codex', generation_hash=fingerprint(row),
                        verdict=verdict, note=note, language_correct=language_correct)
        assert rid not in notes or notes[rid] == decision, 'Preserve prior explicit reading'
        notes[rid] = decision
    write_jsonl(path, list(notes.values()))
    print({'recorded_this_packet': len(decisions), 'total': len(notes), 'stem': args.stem})


if __name__ == '__main__': main()
