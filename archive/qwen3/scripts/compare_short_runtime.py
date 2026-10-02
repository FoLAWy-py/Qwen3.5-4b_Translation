"""Compare complete fixed-input runs without declaring semantic acceptance."""
import argparse
import json
from pathlib import Path
from archive.qwen3.runtime import parse_translation
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def load(path):
    rows = read_jsonl(path)
    summary = json.loads(Path(path).with_suffix('.summary.json').read_text(encoding='utf-8'))
    frozen = read_jsonl('data/prepared/short-benchmark-v1/inputs.jsonl')
    inputs = {r['id']:r['input'] for r in frozen}
    expected = {(i,identifier) for i in (1,2,3) for identifier in inputs}
    keyed = {(r['round'],r['id']):r for r in rows}
    if len(rows)!=81 or len(keyed)!=len(rows) or set(keyed)!=expected:
        raise ValueError('Three complete frozen rounds required')
    if summary['input_hash']!=fingerprint(frozen) or summary['overall']['count']!=len(rows):
        raise ValueError('Summary input identity/count mismatch')
    for row in rows:
        if row['input']!=inputs[row['id']]:
            raise ValueError('Frozen input changed')
        if not row.get('valid_json') or not row.get('ended') or 'error' in row:
            raise ValueError('Non-normal output; preserve evidence and investigate')
        if parse_translation(row['raw'])!=row['prediction']:
            raise ValueError('Raw/parsed prediction mismatch')
    return keyed,summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eager',default='runs/v7-short-eager.jsonl')
    parser.add_argument('--compiled',default='runs/v7-short-compiled.jsonl')
    parser.add_argument('--output',default='runs/v7-short-runtime-comparison.json')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Preserve runtime comparison evidence')
    eager,es = load(args.eager)
    compiled,cs = load(args.compiled)
    for field in ('input_hash','adapter_sha256','prompt_hash','quantization','max_length','max_new_tokens','repetitions'):
        if es[field]!=cs[field]:
            raise ValueError('Runtime comparison identity/settings mismatch')
    differences = []
    raw_matches = 0
    for key in eager:
        a,b = eager[key],compiled[key]
        raw_matches += a['raw']==b['raw']
        if a['prediction']!=b['prediction']:
            differences.append({'id':b['id'],'round':b['round'],'input':b['input'],
                'eager_raw':a['raw'],'compiled_raw':b['raw'],
                'eager_prediction':a['prediction'],'compiled_prediction':b['prediction'],
                'semantic_review':'pending'})
    eo,co = es['overall'],cs['overall']
    report = {'at':now(),'eager':args.eager,'compiled':args.compiled,
        'input_hash':es['input_hash'],'adapter_sha256':es['adapter_sha256'],
        'same_raw_outputs':raw_matches,'same_translation_outputs':81-len(differences),
        'changed_translation_outputs':len(differences),
        'eager_metrics':eo,'compiled_metrics':co,
        'mean_latency_ratio':co['mean_seconds']/eo['mean_seconds'],
        'eager_warm_seconds':es['warm_seconds'],'compiled_warm_seconds':cs['warm_seconds'],
        'eager_peak_reserved_gib':es['peak_reserved_gib'],'compiled_peak_reserved_gib':cs['peak_reserved_gib'],
        'compiled_numerical_performance_passed':cs['performance_passed'],
        'compiler_counters':cs.get('compiler_counters'),
        'deployment_approved':False,'release_approved':False,
        'scope':'Known-input performance comparison only. Output changes require source-grounded reading; exact agreement is consistency evidence, not translation correctness. Different documented warmup counts and cold compile cost retained.'}
    write_jsonl(Path(args.output).with_suffix('.changed.jsonl'),differences)
    write_json(args.output,report)
    print(report,flush=True)


if __name__=='__main__':
    main()
