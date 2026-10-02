"""Blind bounded parallel API authoring from explicit novel plot cards.

No model outputs, historic texts, file locations or retrieved neighbors are sent.
API audit is advisory; individual local Codex review remains mandatory.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import threading
import httpx
from witrans_tools.common import append_jsonl,fingerprint,now,secret,write_json
from scripts.independent_source_pipeline import ask

ROOT=Path('data/independent-20261002-v2-drafts')
AUTHOR='''Write original bilingual translation-test scenes for exactly the supplied plot cards. This is synthetic AI data, not human or sourced prose. Never use a published passage. Preserve each distinctive plot: do not turn everything into access policies or numerical-rule announcements. Return JSON {"scenes":[...]} with one object per card and exactly its id. Each has id, en, zh, context_en, context_zh, multisentence, facts. en/zh must be fully equivalent natural counterparts, preserving tense, roles, negation, scope, conditions, quantities and units. Even-numbered cards: two short sentences, 24-40 English words total. Odd-numbered cards: one natural sentence, 12-24 English words. For indices0-3 include a short genuinely disambiguating context (8-18 English words) and faithful Chinese context; elsewhere contexts may be empty. Context must resolve a word, referent or intent; it must not supply the translation answer, impose missing source facts or merely repeat the sentence. No glossary is requested: do not produce term definitions disguised as required translations. Do not invent real regulations or assert specialist facts as established scientific truths; scenario details are hypothetical. Include a compact list facts naming actual source facts for checking. Vary sentence intentions naturally. Keep scenarios plausible and grammatically clear; resolve misleading specialist wording. Avoid date/number-only paraphrase variants. The two language sentences themselves, without adding context to the translation, must preserve all explicit information.'''
AUDITOR='''Individually audit the supplied newly AI-written bilingual references before any candidate output exists. Return JSON {"reviews":[{"id":...,"approved":bool,"note":...}]} for every id. Be skeptical: reject mismatched entities, tense, roles, negation, scope, quantities, units and conditions; unnatural or erroneous specialist terminology; nonsensical setups; contexts adding missing source facts; and duplicate/near-paraphrase plots within this batch. Do not automatically approve because bilingual wording looks similar. Identify actual facts compared in each note. References need not be word-for-word. This advisory AI audit never substitutes for local source isolation and Codex item-level review.'''

def cards():
    parsed={};category=None
    for line in Path('data/independent-original-scene-cards-20261002.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('['):category=line.strip('[]');parsed[category]=[]
        elif line.strip():parsed[category].extend(line.split('|'))
    if set(parsed)!={'daily','travel','food','academic','hard'} or any(len(v)!=100 for v in parsed.values()):
        raise ValueError('Exactly100 explicit distinct plot cards per category required')
    return [dict(id=f'{"confirmation" if i<40 else "release"}-{category}-{i:03d}',
        stage='confirmation' if i<40 else 'release',category=category,index=i,plot=plot)
        for category,values in parsed.items() for i,plot in enumerate(values)]

def batch_job(batch):
    with httpx.Client(headers={'Authorization':'Bearer '+secret('DEEP_INFRA_APIKEY')},timeout=180) as client:
        authored,metadata=ask(client,AUTHOR,dict(cards=batch),temperature=.65,max_tokens=6500)
        scenes=authored['scenes']
        if len(scenes)!=len(batch) or {s['id'] for s in scenes}!={c['id'] for c in batch}:raise ValueError('Incomplete scene card identities')
        for scene in scenes:
            for key in ('en','zh','context_en','context_zh'):
                if not isinstance(scene.get(key),str):raise ValueError('Missing source/context '+key)
            if type(scene.get('multisentence')) is not bool or not isinstance(scene.get('facts'),list):raise ValueError('Explicit structure required')
        reviewed,auditmeta=ask(client,AUDITOR,dict(scenes=scenes),max_tokens=3500)
        if {r['id'] for r in reviewed['reviews']}!={c['id'] for c in batch}:raise ValueError('Incomplete advisory audit')
        return dict(cards=batch,authored=authored,author_metadata=metadata,advisory_audit=reviewed,
            audit_metadata=auditmeta,candidate_outputs_read=False,local_review='pending',human_data=False)

def main():
    runtime=json.loads(Path('runs/takeover-20261002/runtime-freeze.json').read_text(encoding='utf-8'))
    if not runtime.get('runtime_frozen'):raise ValueError('Unfrozen runtime')
    allcards=cards();ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'batches.jsonl').exists():raise FileExistsError('Preserve existing blind results')
    write_json(ROOT/'plan.json',dict(at=now(),cards_hash=fingerprint(allcards),runtime_hash=fingerprint(runtime),
        author_prompt_hash=fingerprint(AUTHOR),auditor_prompt_hash=fingerprint(AUDITOR),
        source_kind='synthetic_AI',human_data=False,candidate_outputs_read=False,
        parallelism='6 independent CPU/API source batches; no GPU task',
        local_full_source_reference_and_family_review_required=True,release_approved=False))
    completed=0;errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(batch_job,allcards[i:i+10]):i for i in range(0,len(allcards),10)}
        for job in as_completed(jobs):
            index=jobs[job]
            try:
                result=job.result();append_jsonl(ROOT/'batches.jsonl',dict(batch_index=index,**result));completed+=10
            except Exception as exc:
                error=dict(batch_index=index,error=type(exc).__name__+': '+str(exc));errors.append(error);append_jsonl(ROOT/'errors.jsonl',error)
            write_json(ROOT/'progress.json',dict(at=now(),completed_scenes=completed,total_scenes=500,
                errors=errors,candidate_outputs_read=False,status='blind_authoring_advisory_audit_running'))
            print(dict(batch_index=index,completed=completed,errors=len(errors)),flush=True)
    write_json(ROOT/'completed.json',dict(at=now(),scenes=completed,errors=errors,
        candidate_outputs_read=False,status='local_full_review_pending',release_approved=False))

if __name__=='__main__':main()
