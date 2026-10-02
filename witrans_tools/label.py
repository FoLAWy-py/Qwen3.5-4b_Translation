from __future__ import annotations

import json
import time
import concurrent.futures
from pathlib import Path

import httpx

from .protocol import make_messages, parse_translation
from .common import TEACHER_ID, append_jsonl, fingerprint, now, read_jsonl, secret
from .data import check_constraints, validate_record

ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"


def label_one(client, record, max_tokens=512):
    response = client.post(ENDPOINT, json={
        "model": TEACHER_ID, "messages": make_messages(**record["input"]),
        "temperature": 0, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    if response.status_code != 200:
        # Never log request headers or provider bodies containing sensitive values.
        raise RuntimeError(f"DeepInfra HTTP {response.status_code}")
    payload = response.json()
    choice = payload["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("标注输出未正常结束")
    raw = choice["message"]["content"]
    output = parse_translation(raw)
    if not output["translation"].strip():
        raise ValueError("标注译文为空")
    warnings = check_constraints(record["input"]["text"], output["translation"])
    result = {**record, "output": output, "review": {"status": "pending", "warnings": warnings},
              "label_metadata": {"provider": "DeepInfra", "requested_model": TEACHER_ID,
                                 "response_model": payload.get("model"), "at": now(),
                                 "parameters": {"temperature": 0, "max_tokens": max_tokens,
                                                "response_format": "json_object",
                                                "chat_template_kwargs": {"enable_thinking": False}},
                                 "input_hash": fingerprint(record["input"]),
                                 "usage": payload.get("usage", {})}}
    return result


def label(args):
    if args.limit < 1 or args.retries < 0:
        raise ValueError("limit 必须为正数，retries 不能为负")
    rows = read_jsonl(args.input)
    ids = set()
    for row in rows:
        validate_record(row)
        if row["id"] in ids:
            raise ValueError("输入 id 重复")
        ids.add(row["id"])
    completed = {}
    if Path(args.output).exists():
        for row in read_jsonl(args.output):
            validate_record(row, require_output=True)
            if row["id"] in completed:
                raise ValueError("已标注数据 id 重复")
            completed[row["id"]] = fingerprint(row["input"])
    for row in rows:
        if row["id"] in completed and completed[row["id"]] != fingerprint(row["input"]):
            raise ValueError("已标注 id 的输入有变更；请使用新 id")
    todo = [r for r in rows if r["id"] not in completed][:args.limit]
    prompt_tokens = completion_tokens = 0
    succeeded = failed = 0
    workers = getattr(args, "workers", 1)
    if not 1 <= workers <= 16:
        raise ValueError("workers 必须在 1–16 之间")
    with httpx.Client(headers={"Authorization": f"Bearer {secret('DEEP_INFRA_APIKEY')}"}, timeout=120) as client:
        def request(record):
            for attempt in range(args.retries + 1):
                try:
                    result = label_one(client, record, args.max_tokens)
                    return record, result, None
                except (httpx.HTTPError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
                    if isinstance(exc, RuntimeError) and any(str(exc).endswith(str(code)) for code in (400, 401, 403, 404)):
                        raise
                    if attempt == args.retries:
                        return record, None, type(exc).__name__
                    else:
                        time.sleep(min(2 ** attempt, 8))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for record, result, error in pool.map(request, todo):
                # All durable writes happen in this single thread, in input order.
                if error:
                    failed += 1
                    append_jsonl(Path(args.output).with_suffix(".failures.jsonl"), {
                        "id": record["id"], "at": now(), "error_type": error})
                    print(f"{record['id']}: 失败 {error}", flush=True)
                else:
                    append_jsonl(args.output, result)
                    usage = result["label_metadata"]["usage"]
                    prompt_tokens += usage.get("prompt_tokens", 0)
                    completion_tokens += usage.get("completion_tokens", 0)
                    succeeded += 1
                    if workers == 1 or succeeded % 100 == 0 or succeeded + failed == len(todo):
                        print(f"已保存 {succeeded}/{len(todo)}，失败 {failed}，待审核", flush=True)
    print(json.dumps({"saved": succeeded, "failed": failed, "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens}, ensure_ascii=False))
    if failed:
        raise RuntimeError("部分标注失败；重新运行可恢复")
