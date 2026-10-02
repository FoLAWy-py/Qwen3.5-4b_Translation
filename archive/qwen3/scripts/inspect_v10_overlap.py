import argparse
import json
from witrans_tools.common import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--direction', default='zh-CN')
parser.add_argument('--minimum',type=float,default=.25)
args = parser.parse_args()
for row in read_jsonl('runs/v10-source-overlap-candidates.jsonl'):
    if not row['id'].endswith('-'+args.direction):
        continue
    train = [m for m in row['matches']['train'] if m['score']>=args.minimum]
    heldout = [m for m in row['matches']['heldout'] if m['score']>=args.minimum]
    if train or heldout:
        print(json.dumps({'id':row['id'],'train':train[:1],'heldout':heldout[:1]},ensure_ascii=False))
