"""Freeze all development augmentations before candidate output generation."""
import json
import time
from pathlib import Path
from witrans_tools.common import read_jsonl, write_json, write_jsonl, now
from witrans_tools.rag import OpenAIEmbeddings, TerminologyRAG, digest


def main():
    root = Path("runs/takeover-20261002/rag")
    bank = Path("data/rag-reviewed-training-terms-20261002.jsonl")
    sets = {"known": read_jsonl("data/prepared/v12-factorial-v2/dev.jsonl"),
            "public": read_jsonl("data/prepared/v12-factorial-v2/public-dev.jsonl"),
            "performance": read_jsonl("runs/qwen35-v3-diagnostic-recovery1/performance-inputs.jsonl")}
    if [len(sets[k]) for k in sets] != [200, 116, 24]:
        raise ValueError("Expected complete frozen 200/116/24 datasets")
    client = OpenAIEmbeddings(cache_path=".cache/rag/query-embeddings.json")
    semantic = TerminologyRAG(bank, root/"index.json", embeddings=client)
    local = TerminologyRAG(bank, root/"index.json", profile="local_terms")
    query_texts = []
    for rows in sets.values():
        for row in rows:
            inp = row["input"]
            _, trace = local.augment(inp)
            if trace["candidates"]:
                query_texts.append("Translation source: " + inp["text"] +
                                   ("\nUser context: " + inp.get("context", "") if inp.get("context") else ""))
    if query_texts:
        client.embed(list(dict.fromkeys(query_texts)))
    files = {}
    stats = {}
    for name, rows in sets.items():
        prepared = []
        for row in rows:
            a, sa = semantic.augment(row["input"])
            b, sb = local.augment(row["input"])
            prepared.append({**row, "rag_inputs": {"semantic_terms": a, "local_terms": b},
                             "retrieval": {"semantic_terms": sa, "local_terms": sb}})
        path = root/(name + "-prepared.jsonl")
        write_jsonl(path, prepared)
        files[str(path)] = digest(prepared)
        stats[name] = {"rows": len(rows), "semantic_hits": sum(bool(r["retrieval"]["semantic_terms"]["selected"]) for r in prepared),
            "local_hits": sum(bool(r["retrieval"]["local_terms"]["selected"]) for r in prepared),
            "profile_payload_differences": sum(r["rag_inputs"]["semantic_terms"] != r["rag_inputs"]["local_terms"] for r in prepared)}
    # Persistent-client single-query latency, cold query cache. No translation outputs used.
    online = []
    for query in list(dict.fromkeys(query_texts))[:12]:
        before = time.perf_counter()
        client.embed([query], use_cache=False)
        online.append(time.perf_counter()-before)
    client.close()
    write_json(root/"prepared-freeze.json", {"at": now(), "files": files, "stats": stats,
        "bank_hash": digest(semantic.bank), "protocol_hash": digest(json.loads((root/"protocol.json").read_text(encoding="utf-8"))),
        "embedding_events": client.events, "online_uncached_query_seconds": online,
        "quality_preparation_cache": "Batched query embeddings for throughput only; not real-time inference speed evidence",
        "profiles_selected_before_model_outputs": ["semantic_terms", "local_terms"],
        "runtime_freeze": json.loads(Path("runs/takeover-20261002/runtime-freeze.json").read_text(encoding="utf-8")),
        "implementation_hashes": {str(p):digest(p.read_text(encoding="utf-8")) for p in [Path("witrans_tools/rag.py"),Path(__file__)]},
        "independent_test_used": False, "release_approved": False})
    print(stats, flush=True)
    print({"online_uncached_query_seconds": online}, flush=True)


if __name__ == "__main__":
    main()
