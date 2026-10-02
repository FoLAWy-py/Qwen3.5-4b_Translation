"""Bind exactly12 read training audit outputs; do not turn rankings into grades."""
from pathlib import Path
from data.v15_training_output_readings import READINGS
from witrans_tools.common import fingerprint,read_jsonl,write_jsonl


def main():
    path = Path('runs/v15-training-triage-manual.jsonl')
    if path.exists():
        raise ValueError('Preserve existing training audit')
    rows = read_jsonl('runs/v15-training-triage-generations.jsonl')[:12]
    expected = '868dedc95d724d03f216f2a7089f1ea0b90a44b1c68c2794acc8b8da8e16429c'
    if fingerprint(rows)!=expected:
        raise ValueError('Individually read snapshot changed')
    notes = {}
    for line in READINGS.strip().splitlines():
        key,verdict,note = line.split('|',2)
        notes[key]=(verdict,note)
    if set(notes)!={row['id'] for row in rows}:
        raise ValueError('Read IDs changed')
    decisions = [{'id':row['id'],'reviewer':'Codex','generation_hash':fingerprint(row),
                  'verdict':notes[row['id']][0],'note':notes[row['id']][1],'language_correct':True,
                  'reference_status':'accepted','training_contrast_authorized':False,
                  'scope':'TRAIN-only source-based audit; all remaining outputs unread.'} for row in rows]
    write_jsonl(path,decisions)
    print({'individually_read_training_outputs':len(decisions),'new_contrasts_authorized':False})


if __name__=='__main__':
    main()
