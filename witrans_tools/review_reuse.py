"""Reuse a human-authorized review only for exactly identical semantic evidence."""
from .common import fingerprint, now

def content_hash(row):
    return fingerprint({k: row[k] for k in ('input', 'raw', 'reference')})

def indexed(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate evidence IDs')
    return result

def reuse_identical_reviews(previous_outputs, previous_reviews, current_outputs):
    old, decisions, current = map(indexed, (previous_outputs, previous_reviews, current_outputs))
    if set(old) != set(decisions) or set(old) != set(current):
        raise ValueError('Incomplete matching evidence')
    result = []
    for row in current_outputs:
        prior, decision = old[row['id']], decisions[row['id']]
        if decision.get('verdict') not in ('pass', 'minor', 'major', 'critical') or not decision.get('reviewer'):
            raise ValueError('Missing actual review')
        if decision.get('output_hash') != content_hash(prior):
            raise ValueError('Prior decision changed or stale')
        if content_hash(row) != content_hash(prior):
            raise ValueError(f"Current content changed: {row['id']}; inspect anew")
        for field in ('category', 'ended', 'prediction', 'error'):
            if row.get(field) != prior.get(field):
                raise ValueError(f'Current structural evidence changed: {field}')
        if decision.get('ended') != row.get('ended') or decision.get('format_valid') != ('prediction' in row):
            raise ValueError('Review and structural result mismatch')
        result.append({**decision, 'reused_at': now(),
            'reuse_method': 'Exact source/raw/reference and structural equality to individually reviewed prior output; original reviewer and timestamp retained',
            'output_hash': content_hash(row)})
    return result
