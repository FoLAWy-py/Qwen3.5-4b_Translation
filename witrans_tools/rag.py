"""Opt-in, source-bound terminology retrieval; frozen translator stays unchanged."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 3072


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def contains(text: str, phrase: str) -> bool:
    """English whole-word matching, Chinese phrase matching; never fuzzy replacement."""
    if re.search(r"[\u3400-\u9fff]", phrase):
        return phrase in text
    return bool(re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text, re.I))


class OpenAIEmbeddings:
    def __init__(self, env_file=".env", cache_path=None):
        import os
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for semantic retrieval")
        self.client = httpx.Client(base_url="https://api.openai.com/v1/", timeout=60,
                                   headers={"Authorization": "Bearer " + key})
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path and self.cache_path.exists() else {}
        self.events = []

    def close(self):
        self.client.close()

    def embed(self, texts, *, use_cache=True):
        if not texts or any(not isinstance(s, str) or not s.strip() for s in texts):
            raise ValueError("Embeddings require nonempty text strings")
        started = time.perf_counter()
        keys = [digest({"model": EMBEDDING_MODEL, "dimensions": DIMENSIONS, "text": s}) for s in texts]
        missing = list(dict.fromkeys(k for k in keys if not use_cache or k not in self.cache))
        by_key = dict(zip(keys, texts))
        vectors = {}
        usage = 0
        requests = 0
        for offset in range(0, len(missing), 64):
            batch = missing[offset:offset + 64]
            requests += 1
            # An API failure is explicit. No silent lexical/model fallback or leaked error body.
            response = self.client.post("embeddings", json={"model": EMBEDDING_MODEL,
                "input": [by_key[k] for k in batch], "encoding_format": "float"})
            if not response.is_success:
                raise RuntimeError(f"OpenAI embeddings HTTP {response.status_code}; request_id={response.headers.get('x-request-id', 'unavailable')}")
            data = response.json()
            if data.get("model") != EMBEDDING_MODEL:
                raise RuntimeError("Embedding model identity mismatch")
            items = sorted(data["data"], key=lambda d: d["index"])
            if [d["index"] for d in items] != list(range(len(batch))):
                raise RuntimeError("Embedding response indexes/count mismatch")
            for key, item in zip(batch, items):
                vector = item["embedding"]
                if len(vector) != DIMENSIONS or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                    raise RuntimeError("Embedding dimensions or finite-value validation failed")
                norm = math.sqrt(sum(v*v for v in vector))
                if norm == 0:
                    raise RuntimeError("Zero embedding vector")
                vectors[key] = [v/norm for v in vector]
            usage += data["usage"]["total_tokens"]
        if use_cache:
            self.cache.update(vectors)
            if self.cache_path and vectors:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.cache_path.with_suffix(".tmp")
                temp.write_text(json.dumps(self.cache, separators=(",", ":")), encoding="utf-8")
                temp.replace(self.cache_path)
        self.events.append({"seconds": time.perf_counter()-started, "texts": len(texts),
                            "cache_hits": len(keys)-sum(k in missing for k in keys),
                            "requests": requests, "input_tokens": usage, "model": EMBEDDING_MODEL})
        return [vectors[k] if k in vectors else self.cache[k] for k in keys]


class TerminologyRAG:
    """Two preregistered profiles; explicit glossary always takes precedence."""
    def __init__(self, bank_path, index_path, *, profile="semantic_terms", embeddings=None):
        if profile not in ("semantic_terms", "local_terms"):
            raise ValueError("Unsupported RAG profile")
        self.bank = [json.loads(s) for s in Path(bank_path).read_text(encoding="utf-8").splitlines() if s]
        self.index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if self.index["bank_hash"] != digest(self.bank) or self.index["model"] != EMBEDDING_MODEL or self.index["dimensions"] != DIMENSIONS:
            raise ValueError("RAG index identity mismatch")
        if len(self.index["vectors"]) != len(self.bank):
            raise ValueError("RAG index row count mismatch")
        self.profile = profile
        self.embeddings = embeddings
        if profile == "semantic_terms" and embeddings is None:
            raise ValueError("Semantic retrieval requires an embedding client")

    def augment(self, payload, *, use_cache=True):
        started = time.perf_counter()
        text, context = payload["text"], payload.get("context", "")
        target = payload["target_lang"]
        if target not in ("en", "zh-CN"):
            raise ValueError("Unsupported translation direction")
        source_key, target_key = ("zh", "en") if target == "en" else ("en", "zh")
        explicit = dict(payload.get("glossary", {}))
        candidates = []
        for pos, row in enumerate(self.bank):
            term = row[source_key]
            if not contains(text, term) or term in explicit:
                continue
            # A word's other sense must have affirmative source/context evidence.
            anchors = row.get("anchors", [])
            if anchors and not any(contains(text + "\n" + context, a) for a in anchors):
                continue
            candidates.append((pos, row))
        query_seconds = 0.0
        if candidates and self.profile == "semantic_terms":
            query = "Translation source: " + text + ("\nUser context: " + context if context else "")
            before = time.perf_counter()
            vector = self.embeddings.embed([query], use_cache=use_cache)[0]
            query_seconds = time.perf_counter()-before
            scored = [(sum(a*b for a, b in zip(vector, self.index["vectors"][pos])), row)
                      for pos, row in candidates]
            scored = [p for p in scored if p[0] >= 0.25]
        else:
            scored = [(1.0, row) for _, row in candidates]
        scored.sort(key=lambda p: (-len(p[1][source_key]), -p[0], p[1]["id"]))
        selected = []
        for score, row in scored:
            term = row[source_key]
            # Conflicting senses abstain rather than injecting contradictory glossary entries.
            senses = {r[target_key] for _, r in scored if r[source_key] == term}
            if len(senses) > 1:
                continue
            if any(contains(term, chosen["term"]) or contains(chosen["term"], term) for chosen in selected):
                continue
            selected.append({"term": term, "translation": row[target_key], "score": score,
                             "id": row["id"], "source_id": row["source_id"]})
            if len(selected) == 3:
                break
        augmented = {**payload, "glossary": {**{r["term"]: r["translation"] for r in selected}, **explicit}}
        return augmented, {"profile": self.profile, "candidates": len(candidates), "selected": selected,
                           "seconds": time.perf_counter()-started, "embedding_seconds": query_seconds,
                           "query_cache_allowed": use_cache, "payload_hash": digest(augmented)}
