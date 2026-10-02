"""Recover only rejected annotations; retain receipts and explicit AI corrections."""
import httpx

from witrans import make_messages, parse_translation
from witrans_tools.common import TEACHER_ID, append_jsonl, fingerprint, now, read_jsonl, secret
from witrans_tools.label import ENDPOINT


def main():
    inputs = read_jsonl("data/v1_label_inputs.jsonl")
    refs = {r["id"]: r for r in read_jsonl("data/v1_references.jsonl")}
    completed = {r["id"] for r in read_jsonl("data/generated/v1_candidates.jsonl")}
    missing = [r for r in inputs if r["id"] not in completed]
    if len(missing) > 10:
        raise ValueError("本脚本只处理少量格式拒收项")
    with httpx.Client(headers={"Authorization": f"Bearer {secret('DEEP_INFRA_APIKEY')}"}, timeout=120) as client:
        for row in missing:
            parameters = {"temperature": 0, "max_tokens": 512, "response_format": {"type": "json_object"},
                          "chat_template_kwargs": {"enable_thinking": False}}
            response = client.post(ENDPOINT, json={"model": TEACHER_ID, "messages": make_messages(**row["input"]), **parameters})
            if response.status_code != 200:
                raise RuntimeError(f"DeepInfra HTTP {response.status_code}")
            payload = response.json()
            choice = payload["choices"][0]
            raw = choice["message"]["content"]
            receipt = {"id": row["id"], "raw": raw, "finish_reason": choice.get("finish_reason"),
                       "usage": payload.get("usage", {}), "at": now()}
            append_jsonl("data/generated/v1_recovery_receipts.jsonl", receipt)
            corrected = False
            try:
                if choice.get("finish_reason") != "stop":
                    raise ValueError("截断")
                output = parse_translation(raw)
            except ValueError:
                # This is an explicit target correction to an independent accepted
                # reference, not a silent JSON repair or successful teacher claim.
                output = refs[row["id"]]["output"]
                corrected = True
            final = {**row, "output": output, "review": {"status": "pending"}, "label_metadata": {
                "provider": "DeepInfra", "requested_model": TEACHER_ID, "response_model": payload.get("model"),
                "at": now(), "parameters": parameters, "input_hash": fingerprint(row["input"]),
                "usage": payload.get("usage", {}), "teacher_format_rejected": corrected,
                "label_source": "Codex accepted reference after teacher format rejection" if corrected else "teacher final JSON",
                "recovery_receipt_hash": fingerprint(receipt)}}
            append_jsonl("data/generated/v1_candidates.jsonl", final)
            print({"id": row["id"], "raw": raw, "explicit_reference_correction": corrected}, flush=True)


if __name__ == "__main__":
    main()
