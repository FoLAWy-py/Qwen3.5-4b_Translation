"""CPU-only parameter-delta verification; this is not a gradient or semantic test."""
import json
import argparse
import re
import hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file
from witrans_tools.common import now, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',default='v16-span')
    parser.add_argument('--candidate')
    parser.add_argument('--parent',default='models/witrans-4b-v7-critical-cpo')
    args=parser.parse_args()
    assert re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',args.prefix)
    destination=Path(f'runs/{args.prefix}-weight-update-diagnostics.json')
    assert not destination.exists(), 'Preserve diagnostics'
    parent_path=Path(args.parent)/'adapter_model.safetensors'
    with parent_path.open('rb') as stream:
        parent_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    parent=load_file(str(parent_path),device='cpu')
    reports={}
    targets=[(args.prefix,args.candidate)] if args.candidate else [('v15','models/witrans-4b-v15-error-cpo'),('v16','models/witrans-4b-v16-span-cpo')]
    for role,adapter in targets:
        metadata=json.loads((Path(adapter)/'witrans_adapter.json').read_text(encoding='utf-8'))
        if args.parent!='models/witrans-4b-v7-critical-cpo':
            assert metadata['parent_adapter_sha256']==parent_sha, 'Explicit parent must match candidate lineage'
        with (Path(adapter)/'adapter_model.safetensors').open('rb') as stream:
            candidate_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        assert candidate_sha==metadata['adapter_sha256']
        candidate=load_file(str(Path(adapter)/'adapter_model.safetensors'),device='cpu')
        assert set(candidate)==set(parent)
        delta_sq=parent_sq=0.0
        changed=numel=0
        tensors=[]
        for name,value in candidate.items():
            before=parent[name]
            assert value.shape==before.shape and torch.isfinite(value).all()
            difference=value.double()-before.double()
            ds=float(difference.square().sum()); ps=float(before.double().square().sum())
            delta_sq+=ds;parent_sq+=ps;changed+=int(ds>0);numel+=value.numel()
            tensors.append(dict(name=name,delta_l2=ds**.5,parent_l2=ps**.5,max_abs_delta=float(difference.abs().max())))
        reports[role]=dict(adapter_sha256=candidate_sha,parent_adapter_sha256=parent_sha,
                           adapter_tensors=len(candidate),changed_tensors=changed,parameters=numel,
                           delta_l2=delta_sq**.5,parent_l2=parent_sq**.5,relative_delta_l2=(delta_sq/parent_sq)**.5,
                           tensors=tensors)
        metrics=json.loads((Path(adapter)/'metrics.json').read_text(encoding='utf-8'))
        norms=[h['gradient_norm_before_clip'] for h in metrics['history'] if 'gradient_norm_before_clip' in h]
        if norms:
            reports[role]['recorded_gradient_norm_before_clip_range']=[min(norms),max(norms)]
            reports[role]['steps_with_gradient_clipping']=sum(v>1.0 for v in norms)
    write_json(destination,dict(at=now(),parent_adapter=args.parent,parent_adapter_sha256=parent_sha,reports=reports,
        scope='Measured CPU adapter parameter deltas confirm actual updates. No parameter-gradient measurement or quality inference.',release_approved=False))
    print({role:{k:v for k,v in r.items() if k!='tensors'} for role,r in reports.items()})


if __name__=='__main__':
    main()
