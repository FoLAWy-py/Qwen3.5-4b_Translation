"""Paired source-group bootstrap for reviewed semantic pass differences."""
import random
from collections import defaultdict

def paired_pass_interval(candidate, baseline, iterations=5000, seed=42):
    def indexed(rows):
        result = {r['id']: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('Empty or duplicate review IDs')
        if any(r['verdict'] not in ('pass', 'minor', 'major', 'critical') for r in rows):
            raise ValueError('Unreviewed verdict')
        return result
    candidate, baseline = indexed(candidate), indexed(baseline)
    if set(candidate) != set(baseline):
        raise ValueError('Paired review IDs do not match')
    if iterations < 100:
        raise ValueError('At least100 bootstrap replicates required')
    groups = defaultdict(lambda: [0, 0])
    for key, c in candidate.items():
        b = baseline[key]
        if c['group_id'] != b['group_id']:
            raise ValueError('Source group mismatch')
        group = groups[c['group_id']]
        group[0] += int(c['verdict'] == 'pass') - int(b['verdict'] == 'pass')
        group[1] += 1
    values = list(groups.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(v[0] for v in drawn) / sum(v[1] for v in drawn))
    samples.sort()
    return {'pass_difference': sum(v[0] for v in values) / len(candidate),
        'group_bootstrap_95_interval': [samples[int(.025 * iterations)], samples[min(iterations - 1, int(.975 * iterations))]],
        'rows': len(candidate), 'source_groups': len(values), 'iterations': iterations, 'seed': seed,
        'scope': 'Percentile paired bootstrap resampling source groups; interval is diagnostic, not a guarantee of generalization'}
