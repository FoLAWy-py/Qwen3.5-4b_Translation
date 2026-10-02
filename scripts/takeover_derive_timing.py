"""Derived per-input timing table; leaves original raw measurements unchanged."""
import argparse
from collections import defaultdict
import csv
import math
import statistics
from pathlib import Path

from witrans_tools.common import fingerprint,now,read_jsonl,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);rows=read_jsonl(root/'performance.jsonl')
    if len(rows)!=72 or {r['repeat'] for r in rows}!={0,1,2}:raise ValueError('Incomplete formal72')
    groups=defaultdict(list)
    for row in rows:groups[row['id']].append(row)
    if len(groups)!=24 or any(len(v)!=3 for v in groups.values()):raise ValueError('Incomplete24-by3')
    columns=['id','body_tokens','prompt_tokens','output_tokens_min','output_tokens_max','mean_seconds','p95_seconds',
        'output_tokens_per_second','input_output_tokens_per_second','max_reserved_gib','all_eos','all_json_valid']
    with (root/'per-input.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for rid,values in groups.items():
            times=[r['seconds'] for r in values]
            writer.writerow(dict(id=rid,body_tokens=values[0]['body_tokens'],prompt_tokens=values[0]['prompt_tokens'],
                output_tokens_min=min(r['generated_tokens_including_eos'] for r in values),
                output_tokens_max=max(r['generated_tokens_including_eos'] for r in values),
                mean_seconds=statistics.mean(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],
                output_tokens_per_second=sum(r['generated_tokens_including_eos'] for r in values)/sum(times),
                input_output_tokens_per_second=sum(r['prompt_tokens']+r['generated_tokens_including_eos'] for r in values)/sum(times),
                max_reserved_gib=max(r['reserved_gib'] for r in values),all_eos=all(r['ended'] for r in values),
                all_json_valid=all('prediction' in r for r in values)))
    write_json(root/'derived-timing-receipt.json',dict(at=now(),source_hash=fingerprint(rows),
        file='per-input.csv',rows=24,measurements=72,original_measurements_unchanged=True,
        note='Per-input P95 with3 measurements is their maximum; formal global P95 uses all72.'))


if __name__=='__main__':main()
