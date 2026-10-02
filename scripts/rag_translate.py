"""Experimental opt-in CLI; the normal witrans entry remains the frozen v2."""
import argparse
import json
import sys
from pathlib import Path
from witrans_tools.rag import OpenAIEmbeddings, TerminologyRAG


def main():
    parser = argparse.ArgumentParser(description="Experimental source-bound terminology RAG")
    parser.add_argument("text")
    parser.add_argument("target_lang", choices=["en", "zh-CN"])
    parser.add_argument("--context", default="")
    parser.add_argument("--glossary-json", default="{}")
    parser.add_argument("--profile", choices=["local_terms", "semantic_terms"], default="local_terms")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--query-cache", action="store_true", help="Opt-in persistent source-text embeddings cache; repeated inputs only")
    args = parser.parse_args()
    client = OpenAIEmbeddings(cache_path=".cache/rag/user-query-embeddings.json" if args.query_cache else None) if args.profile=="semantic_terms" else None
    try:
        rag = TerminologyRAG("data/rag-reviewed-training-terms-20261002.jsonl", "runs/takeover-20261002/rag/index.json",
                            profile=args.profile, embeddings=client)
        payload, trace = rag.augment({"text":args.text,"target_lang":args.target_lang,
            "context":args.context,"glossary":json.loads(args.glossary_json)},use_cache=args.query_cache)
        from witrans_tools.qwen35 import Qwen35Translator
        from witrans_tools.protocol import parse_translation
        translator = Qwen35Translator(runtime="decode_compiled",cache_dir=".cache/qwen35-rag")
        raw, ended = translator.generate_raw(**payload,max_new_tokens=128)
        if not ended:
            raise RuntimeError("Translation did not end normally within the output budget")
        print(json.dumps(parse_translation(raw),ensure_ascii=False))
        if args.trace:
            print(json.dumps({"retrieval":trace,"generation":translator.last_generation_stats},ensure_ascii=False),file=sys.stderr)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
