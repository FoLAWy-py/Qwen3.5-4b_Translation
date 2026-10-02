"""Fixed24, whole-round warmup and three measured rounds, including real retrieval."""
import json
import math
import statistics
import time
from pathlib import Path
import psutil
from witrans_tools.common import read_jsonl, append_jsonl, write_json, now
from witrans_tools.protocol import parse_translation
from witrans_tools.rag import OpenAIEmbeddings, TerminologyRAG, digest


def main():
    root = Path("runs/takeover-20261002/rag")
    destination = root/"performance"
    if destination.exists():
        raise FileExistsError(destination)
    own = psutil.Process()
    allowed = {own.pid, *(p.pid for p in own.parents())}
    inventory = []
    for process in psutil.process_iter(["pid","name","cmdline","create_time"]):
        if "python" in (process.info["name"] or "").lower():
            inventory.append(process.info)
            if process.pid not in allowed:
                raise RuntimeError("Serial GPU prerequisite: " + str(process.info))
    rows = read_jsonl(root/"performance-prepared.jsonl")
    original = [{k:v for k,v in r.items() if k not in ("rag_inputs","retrieval")} for r in rows]
    if len(rows) != 24 or digest(original) != "f4057624ca5a024d3bf286b0d0acb37fcacfd6273250ae9979535d64c7c570f9":
        raise ValueError("Fixed24 identity mismatch")
    freeze = json.loads((root/"prepared-freeze.json").read_text(encoding="utf-8"))
    if digest(rows) != freeze["files"][str(root/"performance-prepared.jsonl")]:
        raise ValueError("RAG input freeze mismatch")
    destination.mkdir()
    write_json(destination/"plan.json", {"at":now(),"processes":inventory,"pid":own.pid,
        "created":own.create_time(),"prepared_freeze_hash":digest(freeze),"fixed24_hash":digest(original),
        "runtime":"decode_compiled","nf4":True,"lora_precision":"float32","compute_dtype":"bfloat16",
        "cache":"static1024","max_length":1024,"max_new_tokens":128,"do_sample":False,"thinking":False,
        "warmup":"Entire24 for each profile; compile cost included; existing compiler cache reused explicitly",
        "measured_rounds":3,"profiles":["local_terms","semantic_terms"],
        "query_embedding_cache":"Disabled in all formal semantic-profile requests, including warmup",
        "order":"Per-input paired; order alternates by round/index",
        "semantic_guard":"Actual retrieval payload must equal preregistered prepared payload; changed retrieval aborts",
        "release_approved":False})
    before = time.perf_counter()
    client = OpenAIEmbeddings()
    rags = {p:TerminologyRAG("data/rag-reviewed-training-terms-20261002.jsonl",root/"index.json",
                            profile=p,embeddings=client if p=="semantic_terms" else None)
            for p in ("local_terms","semantic_terms")}
    index_load_seconds = time.perf_counter()-before
    import torch
    from witrans_tools.qwen35 import Qwen35Translator, validate_placement
    state = {"at":now(),"pid":own.pid,"status":"loading"}
    write_json(destination/"progress.json",state)
    torch.cuda.reset_peak_memory_stats()
    before = time.perf_counter()
    translator = Qwen35Translator(runtime="decode_compiled",cache_dir="runs/takeover-20261002/decode-final")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-before
    completed = {p:[] for p in rags}
    warmups = {}
    try:
        warm_start = time.perf_counter()
        for repeat in (-1,0,1,2):
            for index, ref in enumerate(rows):
                order = list(rags)
                if (repeat+index)%2:
                    order.reverse()
                for profile in order:
                    body = len(translator.tokenizer.encode(ref["input"]["text"],add_special_tokens=False))
                    if not 20 <= body <= 120:
                        raise ValueError("Formal source body token budget violated")
                    torch.cuda.synchronize(); before = time.perf_counter()
                    payload, trace = rags[profile].augment(ref["input"],use_cache=False)
                    if payload != ref["rag_inputs"][profile]:
                        raise RuntimeError("Fresh retrieval changed preregistered payload")
                    generation_start = time.perf_counter()
                    raw, ended = translator.generate_raw(**payload,max_new_tokens=128)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-before
                    row = {"id":ref["id"],"repeat":repeat,"profile":profile,"original_input":ref["input"],
                        "input":payload,"raw":raw,"prediction":parse_translation(raw),"ended":ended,
                        "seconds":seconds,"generation_seconds":time.perf_counter()-generation_start,
                        "retrieval":trace,"body_tokens":body,**translator.last_generation_stats,
                        "reserved_gib":torch.cuda.memory_reserved()/1024**3,"allocated_gib":torch.cuda.memory_allocated()/1024**3}
                    row["end_to_end_output_tokens_per_second"] = row["generated_tokens_including_eos"]/seconds
                    if not ended or row["generated_tokens_including_eos"]>128 or row["prompt_tokens"]+128>1024:
                        raise RuntimeError("Formal token budget or normal EOS check failed")
                    validate_placement(translator.model)
                    append_jsonl(destination/("warmup.jsonl" if repeat==-1 else "performance.jsonl"),row)
                    if repeat>=0:
                        completed[profile].append(row)
                    else:
                        warmups[profile]=warmups.get(profile,0)+seconds
                    state.update(status="running",repeat=repeat,profile=profile,completed=index+1,last_id=ref["id"],at=now())
                    write_json(destination/"progress.json",state)
                    print({"repeat":repeat,"profile":profile,"row":index+1,"seconds":seconds},flush=True)
            if repeat==-1:
                whole_warmup_seconds=time.perf_counter()-warm_start
        summaries={}
        for profile, results in completed.items():
            times=[r["seconds"] for r in results]
            p95=sorted(times)[math.ceil(.95*len(times))-1]
            summaries[profile]={"calls":len(results),"mean_seconds":statistics.mean(times),"p95_seconds":p95,
                "mean_generation_seconds":statistics.mean(r["generation_seconds"] for r in results),
                "mean_retrieval_seconds":statistics.mean(r["retrieval"]["seconds"] for r in results),
                "retrieval_hit_calls":sum(bool(r["retrieval"]["selected"]) for r in results),
                "warmup_seconds_including_compile":warmups[profile],
                "end_to_end_output_tokens_per_second":sum(r["generated_tokens_including_eos"] for r in results)/sum(times),
                "generated_tokens":sum(r["generated_tokens_including_eos"] for r in results),
                "min_max_prompt_tokens":[min(r["prompt_tokens"] for r in results),max(r["prompt_tokens"] for r in results)],
                "outputs_hash":digest(results),"all_json_valid":True,"all_eos":True,
                "performance_passed":statistics.mean(times)<=4 and p95<=8}
        write_json(destination/"summary.json",{"at":now(),"profiles":summaries,"index_load_seconds":index_load_seconds,
            "load_seconds":load_seconds,"whole_warmup_seconds":whole_warmup_seconds,
            "compile_cache":"Previously populated decode-final cache; first compile cost remains in original baseline receipt",
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
            "cpu_parameter_count":0,"adapter_sha256":translator.adapter_sha256,"embedding_events":client.events,
            "semantic_review":"pending","release_approved":False})
        state.update(status="complete")
    except BaseException as exc:
        state.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        client.close();state["at"]=now();write_json(destination/"progress.json",state)


if __name__ == "__main__":
    main()
