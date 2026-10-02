"""Bind explicit earlier readings of exact TRAIN/performance overlap; no auto grading."""
import json
from pathlib import Path

from scripts.review_v12_factorial import binding
from witrans import parse_translation
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl
from witrans_tools.data import validate_record


def main():
    dest=Path('runs/qwen35-v3-performance-train-overlap-manual.jsonl')
    assert not dest.exists(), 'Preserve earlier source reading'
    train=read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    manifest=json.loads(Path('data/prepared/v16-reviewed-families/resolved-manifest.json').read_text(encoding='utf-8'))
    assert fingerprint(train)==manifest['train_ready_hash']
    refs={r['id']:r for r in train}
    rows=read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance.jsonl')
    originals={r['id']:r for r in rows if r['repeat']==0 and r['id'] in refs}
    readings=[]
    for line in Path('runs/qwen35-v3-performance-train-overlap-readings.tsv').read_text(encoding='utf-8').splitlines():
        rid,verdict,language,note=line.split('\t',3)
        ref=refs[rid];row=originals[rid]
        validate_record(ref,True,True,purpose='training')
        assert row['input']==ref['input'] and row['ended'] and row['valid_json']
        normalized=dict(id=rid,category=ref['category'],input=row['input'],reference=ref['output'],
                        raw=row['raw'],ended=row['ended'],prediction=parse_translation(row['raw']))
        readings.append(dict(id=rid,group_id=ref['group_id'],category=ref['category'],target_lang=ref['input']['target_lang'],
            verdict=verdict,language_correct=language=='true',note=note,reviewer='Codex',at=now(),
            format_valid=True,ended=True,binding_hash=binding(normalized),
            source_row_hash=fingerprint(row),chosen_content_hash=ref['review']['content_hash'],
            normalized_output=normalized,source_path='runs/qwen35-v3-diagnostic-recovery1/performance.jsonl',
            adapter_sha256='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27',
            decoding=dict(max_new_tokens=128,max_length=1024,do_sample=False),
            scope='Actual current-model TRAIN source output seen in formal timing. Not DEV-derived or independent. Later discovery reuse requires exact input/raw/reference/prediction/EOS binding; otherwise reread.'))
    assert len(readings)==len({r['id'] for r in readings})==len(originals)==8
    write_jsonl(dest,readings)
    print({'reviewed_train_overlap':len(readings),'major':sum(r['verdict']=='major' for r in readings)},flush=True)


if __name__=='__main__':
    main()
