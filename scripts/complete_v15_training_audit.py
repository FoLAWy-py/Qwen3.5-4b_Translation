"""Export52 explicit readings and verify all64 source-bound TRAIN judgments."""
import json
from collections import Counter
from pathlib import Path
from data.v15_training_output_readings import REMAINING_READINGS
from scripts.export_v13_first_readings import save
from scripts.review_v12_factorial import accept_manual
from witrans_tools.common import fingerprint,now,read_jsonl,write_json


def main():
    plan = json.loads(Path('runs/v15-training-triage-decoding-plan.json').read_text(encoding='utf-8'))
    references = read_jsonl(plan['input'])
    rows = read_jsonl(plan['output'])
    if fingerprint(references)!=plan['data_hash'] or len(rows)!=64 or len(references)!=64:
        raise ValueError('TRAIN audit sources changed')
    if fingerprint(rows[12:])!='f536d7faed93db71a8b9ed116ee30102eae27a64ce9f840d74ad7d4a0ff3a02c':
        raise ValueError('Explicitly read52 output snapshot changed')
    notes = {}
    for line in REMAINING_READINGS.strip().splitlines():
        row_id,verdict,note = line.split('|',2)
        notes[row_id]=(verdict,note)
    if set(notes)!={row['id'] for row in rows[12:]}:
        raise ValueError('Read IDs do not match')
    decisions = [{'id':row['id'],'reviewer':'Codex','generation_hash':fingerprint(row),
                  'verdict':notes[row['id']][0],'note':notes[row['id']][1],'language_correct':True,
                  'reference_status':'accepted','training_contrast_authorized':False,
                  'scope':'TRAIN-only source-based audit; not heldout accuracy.'} for row in rows[12:]]
    save('runs/v15-training-triage-manual.jsonl',decisions)
    manual = {row['id']:row for row in read_jsonl('runs/v15-training-triage-manual.jsonl')}
    refs = {row['id']:row for row in references}
    if len(manual)!=64 or {row['id'] for row in rows}!=set(manual) or set(manual)!=set(refs):
        raise ValueError('Complete64 individual readings required')
    for row in rows:
        accept_manual(row,manual[row['id']])
        ref = refs[row['id']]
        if row['input']!=ref['input'] or row['reference']!=ref['output']:
            raise ValueError('Source/ref no longer binds reading')
    summary = {'at':now(),'rows':64,'source_groups':len({r['group_id'] for r in references}),
               'verdicts':dict(Counter(row['verdict'] for row in manual.values())),
               'complete':True,'new_contrasts_authorized':False,'release_approved':False,
               'scope':'TRAIN-only triage, biased high-NLL selection, not model acceptance or generalization.'}
    write_json('runs/v15-training-triage-manual.summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
