import httpx
import pytest

from witrans_tools.label import label_one


def candidate():
    return {"id": "1", "input": {"text": "Send {user_name} a reminder at 09:30.", "target_lang": "zh-CN"}}


def response(content, finish_reason="stop"):
    return {"model": "Qwen/Qwen3-32B", "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_teacher_output_requires_review():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json=response('{"translation":"在09:30提醒{user_name}。"}'))))
    result = label_one(client, candidate())
    assert result["review"]["status"] == "pending"
    assert result["review"]["warnings"] == []
    assert result["label_metadata"]["usage"]["completion_tokens"] == 5


def test_truncated_teacher_answer_rejected_even_with_valid_json():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json=response('{"translation":"valid-looking"}', "length"))))
    with pytest.raises(ValueError, match="正常结束"):
        label_one(client, candidate())
