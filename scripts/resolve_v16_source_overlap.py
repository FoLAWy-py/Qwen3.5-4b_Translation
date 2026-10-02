"""Explicit reading resolves template similarity; original quarantine stays preserved."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

NOTES={
    'v5-seed-18':'TRAIN cast为手腕石膏拆除或戏剧演员撤下，DEV issue为有印刷错误杂志期号召回；对象、动作及语境不同，只共享The ... has been ...句式，不是同原文或近似改写。',
    'v7-context-09':'TRAIN mole为花园鼹鼠/组织内间谍难发现，DEV chord为吉他和弦难按；对象、动作及语境不同，只共享is difficult to句式，不是同来源或近似改写。',
}


def main():
    root=Path('data/prepared/v16-reviewed-families')
    assert not (root/'train-ready-after-overlap-review.jsonl').exists(), 'Preserve resolved corpus'
    retrieval=json.loads((root/'overlap-retrieval.json').read_text(encoding='utf-8'))
    rows=read_jsonl(root/'all-reviewed.jsonl')
    heldout={r['id']:r for r in read_jsonl('data/prepared/v12-factorial-v2/dev.jsonl')}
    assert {h['train_group'] for h in retrieval['hits']}==set(NOTES)
    by_id={r['id']:r for r in rows}
    decisions=[]
    for hit in retrieval['hits']:
        train,dev=by_id[hit['train_id']],heldout[hit['heldout_id']]
        decisions.append(dict(train_id=train['id'],heldout_id=dev['id'],train_group=train['group_id'],
                              train_input_hash=fingerprint(train['input']),heldout_input_hash=fingerprint(dev['input']),
                              reviewer='Codex',at=now(),decision='distinct_sources_release_train_family',
                              note=NOTES[train['group_id']]))
    assert len(decisions)==4
    write_json(root/'source-overlap-manual-resolution.json',dict(at=now(),retrieval_hash=fingerprint(retrieval),
        decisions=decisions,scope='Source-only comparison, no DEV answers used for training or authoring; original quarantine and manifests retained.'))
    write_jsonl(root/'train-ready-after-overlap-review.jsonl',rows)
    original=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    write_json(root/'resolved-manifest.json',{**original,'at':now(),'supersedes_for_train_ready_only':'manifest.json',
        'train_ready_path':str(root/'train-ready-after-overlap-review.jsonl'),'train_ready_hash':fingerprint(rows),
        'train_ready_families':200,'train_ready_rows':416,'quarantined_families':[],
        'overlap_resolution_hash':fingerprint(decisions)})
    print({'train_ready_families':200,'train_ready_rows':416,'template_hits_explicitly_resolved':4})


if __name__=='__main__':
    main()
