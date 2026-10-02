"""Freeze randomly ordered pair packets with model names withheld from reviewers."""
import secrets
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    packet_path = Path('runs/v12-blind-packets.jsonl')
    key_path = Path('runs/v12-blind-identity-key.json')
    if packet_path.exists() or key_path.exists():
        raise ValueError('Preserve frozen blinded ordering')
    refs = read_jsonl('data/prepared/v4/dev.jsonl')
    models = {role: {row['id']:row for row in read_jsonl(path)} for role,path in
              [('base','runs/v12-base-dev.jsonl'),('v7','runs/v7-critical-dev.jsonl')]}
    packets, identities = [], {}
    for ref in refs:
        order = ['base','v7'] if secrets.randbits(1) else ['v7','base']
        packet_id = secrets.token_hex(12)
        packet = {'id':packet_id, 'input':ref['input'],
                  'outputs':{label:models[role][ref['id']]['raw'] for label,role in zip(('A','B'),order)}}
        for role in order:
            row = models[role][ref['id']]
            if row['input'] != ref['input'] or row['reference'] != ref['output']:
                raise ValueError('Blinded packet source differs')
        packets.append(packet)
        identities[packet_id] = {'source_id':ref['id'], 'group_id':ref['group_id'],
                                 'A':order[0], 'B':order[1], 'packet_hash':fingerprint(packet)}
    secrets.SystemRandom().shuffle(packets)
    write_jsonl(packet_path, packets)
    write_json(key_path, {'at':now(), 'packets_hash':fingerprint(packets), 'mapping':identities,
                         'instruction':'Do not inspect identity map before content-bound decisions are frozen.'})
    print({'blinded_pair_packets':len(packets), 'packets_hash':fingerprint(packets),
           'identity_labels_withheld':True,
           'limit':'Same Codex evaluator has prior knowledge of outputs; masking names is not an independent evaluator or erasure of prior knowledge.'})


if __name__ == '__main__':
    main()
