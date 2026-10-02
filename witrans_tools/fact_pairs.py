"""Validate reviewed source-conditioned pairs before constructing contrastive inputs."""
from .common import fingerprint
from .data import encode_example, validate_record


def validate_fact_packet(packet):
    inputs, outputs = packet.get('inputs',[]), packet.get('outputs',[])
    if len(inputs)!=2 or len(outputs)!=2:
        raise ValueError('Exactly two reviewed inputs and answers required')
    if inputs[0]['text']==inputs[1]['text'] or outputs[0]['translation']==outputs[1]['translation']:
        raise ValueError('Fact contrast must change both source and translation')
    if any(inputs[0].get(key)!=inputs[1].get(key) for key in ('target_lang','context','glossary')):
        raise ValueError('Only source fact may change; direction/context/glossary must match')
    review = packet.get('review',{})
    if (review.get('status')!='approved' or not review.get('reviewer') or not review.get('notes')
            or not review.get('cross_pair_check') or not packet.get('fact_axis')
            or review.get('content_hash')!=fingerprint({'inputs':inputs,'outputs':outputs})):
        raise ValueError('Fact pair lacks bound individual acceptance')
    for index in range(2):
        record = {'id':packet['id']+f'-{index}','group_id':packet['group_id'],'category':packet['category'],
                  'input':inputs[index],'output':outputs[index],'source':packet['source']}
        validate_record(record,require_output=True,require_review=False,purpose='training')
    return packet


def encode_fact_packet(tokenizer, packet, max_length=1024):
    validate_fact_packet(packet)
    matrix = []
    for i in range(2):
        row = []
        for j in range(2):
            record = {'id':packet['id']+f'-{i}{j}', 'input':packet['inputs'][i],
                      'output':packet['outputs'][j]}
            row.append(encode_example(tokenizer,record,max_length))
        matrix.append(row)
    counts = [[sum(value!=-100 for value in row['labels']) for row in source] for source in matrix]
    if counts[0]!=counts[1]:
        raise ValueError('Fixed candidate answer token count changed across source rows')
    return matrix,counts[0]
