"""Serial complete old-set RAG generation; semantic grading is a separate AI read."""
import hashlib
import json
import time
from pathlib import Path
import psutil
from witrans_tools.common import read_jsonl, append_jsonl, write_json, now
from witrans_tools.rag import digest
from witrans_tools.protocol import parse_translation


def main():
    root = Path("runs/takeover-20261002/rag")
    freeze = json.loads((root/"prepared-freeze.json").read_text(encoding="utf-8"))
    for path, checksum in freeze["files"].items():
        if digest(read_jsonl(path)) != checksum:
            raise ValueError("Prepared input freeze mismatch: " + path)
    own = psutil.Process()
    allowed = {own.pid, *(p.pid for p in own.parents())}
    processes = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        if "python" in (p.info["name"] or "").lower():
            processes.append(p.info)
            if p.pid not in allowed:
                raise RuntimeError("Serial GPU prerequisite: " + str(p.info))
    destination = root/"quality"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    write_json(destination/"plan.json", {"at":now(), "pid":own.pid, "created":own.create_time(),
        "processes":processes, "prepared_freeze_hash":digest(freeze),
        "runtime":"decode_compiled", "max_new_tokens":256,"max_length":1024,
        "same_payload_reuse":"Local profile uses the directly generated semantic-profile row only when complete model input is byte-identical. No baseline predictions or references used to generate outputs.",
        "source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "semantic_review":"pending"})
    import torch
    from witrans_tools.qwen35 import Qwen35Translator, validate_placement
    state = {"at":now(), "pid":own.pid, "status":"loading"}
    write_json(destination/"progress.json", state)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    translator = Qwen35Translator(runtime="decode_compiled", cache_dir="runs/takeover-20261002/decode-final")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    summaries = {}
    try:
        for label in ("known", "public"):
            rows = read_jsonl(root/(label+"-prepared.jsonl"))
            counts = {}
            for profile in ("semantic_terms", "local_terms"):
                completed = []
                semantic_rows = read_jsonl(destination/(label+"-semantic_terms.jsonl")) if profile=="local_terms" else []
                for i, ref in enumerate(rows):
                    inp = ref["rag_inputs"][profile]
                    reused = profile=="local_terms" and inp==ref["rag_inputs"]["semantic_terms"]
                    if reused:
                        generated = {k:v for k,v in semantic_rows[i].items() if k not in ("profile","input","retrieval","generation_evidence")}
                        generated["generation_evidence"] = {"reuse_same_complete_payload": True,
                            "direct_file":label+"-semantic_terms.jsonl", "direct_row_id":ref["id"], "payload_hash":digest(inp)}
                    else:
                        torch.cuda.synchronize(); before = time.perf_counter()
                        raw, ended = translator.generate_raw(**inp, max_new_tokens=256)
                        torch.cuda.synchronize()
                        generated = {"id":ref["id"],"group_id":ref["group_id"],"category":ref["category"],
                            "original_input":ref["input"],"reference":ref["output"], "raw":raw,"ended":ended,
                            "seconds":time.perf_counter()-before,**translator.last_generation_stats,
                            "generation_evidence":{"reuse_same_complete_payload":False,"payload_hash":digest(inp)}}
                        try:
                            generated.update(prediction=parse_translation(raw),json_valid=True)
                        except ValueError as exc:
                            generated.update(json_valid=False,error=str(exc))
                    generated.update(profile=profile,input=inp,retrieval=ref["retrieval"][profile],semantic_review="pending")
                    append_jsonl(destination/(label+"-"+profile+".jsonl"), generated)
                    completed.append(generated)
                    state.update(status="running",set=label,profile=profile,completed=i+1,last_id=ref["id"],at=now())
                    write_json(destination/"progress.json",state)
                    print({"set":label,"profile":profile,"row":i+1,"id":ref["id"],"reuse":reused},flush=True)
                counts[profile] = {"rows":len(completed),"json_valid":sum(r["json_valid"] for r in completed),
                    "eos":sum(r["ended"] for r in completed),"direct_generated":sum(not r["generation_evidence"]["reuse_same_complete_payload"] for r in completed),
                    "outputs_hash":digest(completed),"semantic_review":"pending"}
            summaries[label]=counts
        validate_placement(translator.model)
        write_json(destination/"generation-summary.json", {"at":now(),"sets":summaries,"load_seconds":load_seconds,
            "adapter_sha256":translator.adapter_sha256,"cpu_parameter_count":0,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"semantic_review":"pending","release_approved":False})
        state.update(status="generation_complete_review_pending")
    except BaseException as exc:
        state.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        state["at"]=now();write_json(destination/"progress.json",state)


if __name__ == "__main__":
    main()
