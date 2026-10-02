"""Bind root Codex AI's completed whole-set source/context/output reading."""
from pathlib import Path
from witrans_tools.common import read_jsonl, write_jsonl, write_json, now
from witrans_tools.rag import digest
from witrans_tools.independent_quality import counts, group_intervals
from witrans_tools.paired_stats import paired_pass_interval

ROOT=Path("runs/takeover-20261002/rag/quality")
BASE=Path("runs/takeover-20261002/development-final")
# Manually adjudicated actual changes, after reading each source and prediction.
CHANGES={
    "v4-dev-039-en":("pass","adjoining rooms 与原文相邻房间等义，房间之间无连通门的否定完整；原译 adjacent rooms 也正确。"),
    "v4-dev-041-zh-CN":("pass","孜然籽短时烘烤后再研磨，术语和时序完整；没有改变此前 pass 等级。"),
    "v4-dev-041-en":("pass","先短时烘炒孜然籽再研磨，Toast 祈使与中文烹饪步骤一致；原译同样合格。"),
    "v4-dev-056-zh-CN":("minor","无气泡水比普通水更明确，但瓶子是无气泡水仍为不自然的对象表达，不能升为 pass。"),
    "v4-dev-077-zh-CN":("pass","对照组接受相同指示但无额外练习，角色和否定保留；instructions 的指示与原译指导均合理。"),
    "v4-dev-080-en":("pass","加入溶剂稀释样品，同时保留全部溶质，溶剂与溶质角色未颠倒；与原译同样合格。"),
}


def main():
    # This receipt is written only after the root has read every actual output.
    receipt={"at":now(),"reviewer":"Codex AI","human_acceptance":False,
        "method":"Root AI individually read all316 actual original source/context/reference/RAG predictions. Source-grounded prior judgments retained only after rereading; all changed raws separately adjudicated. JSON/EOS/identity checks are not semantic scoring.",
        "same_payload_profile_audit":"All316 semantic and local profiles have identical complete model payloads and actual raws. One full source/context/output read applies to both; local generation evidence transparently links identical-payload reuse.",
        "new_independent_tests_read_for_this_rag_experiment":False,"release_approved":False,"sets":{}}
    seen=set()
    for label,expected in (("known",200),("public",116)):
        original=read_jsonl(BASE/(label+".jsonl"))
        prior=read_jsonl(BASE/(label+"-semantic.jsonl"))
        profiles={p:read_jsonl(ROOT/(label+"-"+p+".jsonl")) for p in ("semantic_terms","local_terms")}
        if len(original)!=expected or any(len(rows)!=expected for rows in profiles.values()):
            raise ValueError("Full-set coverage is required before semantic binding")
        a,b=profiles.values()
        if any(x['id']!=y['id'] or x['input']!=y['input'] or x['raw']!=y['raw'] for x,y in zip(a,b)):
            raise ValueError("Separate profile adjudication required")
        reports={}
        for profile,actual in profiles.items():
            reviews=[];changes=[]
            for ref,old,row in zip(original,prior,actual):
                if ref['id']!=row['id'] or old['id']!=row['id'] or ref['input']!=row['original_input']:
                    raise ValueError("Source-bound review identity mismatch")
                changed=ref['raw']!=row['raw']
                if changed and row['id'] not in CHANGES:
                    raise ValueError("Actual change needs explicit root adjudication: "+row['id'])
                verdict,note=CHANGES[row['id']] if changed else (old['verdict'],old['note'])
                if changed:
                    seen.add(row['id']);changes.append({"id":row['id'],"before":old['verdict'],"after":verdict,"note":note})
                review={**old,"verdict":verdict,"note":note,"json_valid":row['json_valid'],"ended":row['ended'],
                    "source_input_hash":digest(row['original_input']),"input_hash":digest(row['input']),
                    "output_hash":digest(row['raw']),"generation_hash":digest(row),"profile":profile,
                    "retrieved_terms_count":len(row['retrieval']['selected']),"retrieved_terms_preserved":True,
                    "review_source":"complete_actual_source_context_prediction_reading_RAG",
                    "reviewer":"Codex AI","human_acceptance":False}
                reviews.append(review)
            write_jsonl(ROOT/(label+"-"+profile+"-semantic.jsonl"),reviews)
            reports[profile]={"counts":counts(reviews),"source_group_intervals":group_intervals(reviews),
                "baseline_counts_same_source_review":counts(prior),
                "paired_pass_change":paired_pass_interval(reviews,prior,iterations=10000,seed=20261002),
                "changed_raw_count":len(changes),"changed_rows":changes,
                "new_major":sum(r['verdict'] in ('major','critical') and p['verdict'] not in ('major','critical') for r,p in zip(reviews,prior)),
                "new_language_errors":sum(not r['language_correct'] and p['language_correct'] for r,p in zip(reviews,prior)),
                "retrieval_hits":sum(bool(r['retrieved_terms_count']) for r in reviews),
                "strata":{c+'/'+d:counts([r for r in reviews if r['category']==c and r['target_lang']==d]) for c in sorted({r['category'] for r in reviews}) for d in ('en','zh-CN')},
                "source_constraints":counts([r for r in reviews if r['has_constraint']]),
                "constraint_retained":sum(r['constraint_preserved'] for r in reviews if r['has_constraint']),
                "actual_outputs_hash":digest(actual),"reviews_hash":digest(reviews),"semantic_review_complete":True}
        receipt['sets'][label]=reports
    if seen!=set(CHANGES):
        raise ValueError("Adjudication table does not match actual changed outputs")
    receipt.update(full316_review_complete=True,net_pass_gain=0,major_reduction=0,
        adoption_quality_gain_gate_passed=False,
        decision="Retain frozen no-RAG default. This177-term experimental bank has no observed net semantic grade improvement; no independent RAG validation or release claim.")
    write_json(ROOT/"semantic-summary.json",receipt)
    print({"complete":True,"net_pass_gain":0,"major_reduction":0,"adopt_default":False})


if __name__=="__main__":main()
