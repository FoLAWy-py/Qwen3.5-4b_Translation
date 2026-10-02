import json

import httpx
import pytest

from witrans_tools.rag import DIMENSIONS, EMBEDDING_MODEL, OpenAIEmbeddings, TerminologyRAG, contains, digest


def bank(tmp_path, rows):
    path = tmp_path / "bank.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"bank_hash": digest(rows), "model": EMBEDDING_MODEL,
        "dimensions": DIMENSIONS, "vectors": [[1.0]+[0.0]*(DIMENSIONS-1) for _ in rows]}), encoding="utf-8")
    return path, index


def term(id, en, zh, anchors=()):
    return {"id":id,"en":en,"zh":zh,"anchors":list(anchors),"source_id":"approved-training"}


def test_word_boundaries_prevent_accidental_term_injection():
    assert not contains("thanksgiving", "giving")
    assert not contains("the claimant", "claim")
    assert contains("The Eigenvalue is zero.", "eigenvalue")
    assert contains("保留特征值", "特征值")


def test_ambiguous_terms_require_evidence_and_conflicts_abstain(tmp_path):
    paths = bank(tmp_path, [term("1","bank","河岸",["river"]), term("2","bank","银行",["account"])])
    rag = TerminologyRAG(*paths, profile="local_terms")
    def lookup(text):
        return rag.augment({"text":text,"target_lang":"zh-CN"})[0]["glossary"]
    assert lookup("The bank is closed.") == {}
    assert lookup("The bank of the river.") == {"bank":"河岸"}
    assert lookup("The river bank holds an account.") == {}


def test_user_glossary_wins_and_no_hit_never_calls_api(tmp_path):
    paths = bank(tmp_path, [term("1","eigenvalue","特征值")])
    class Never:
        def embed(self, *args, **kwargs):
            raise RuntimeError("Unexpected network call")
    rag = TerminologyRAG(*paths, embeddings=Never())
    explicit = {"eigenvalue":"特征根"}
    result, trace = rag.augment({"text":"the eigenvalue","target_lang":"zh-CN","glossary":explicit})
    assert result["glossary"] == explicit
    assert trace["candidates"] == 0
    assert rag.augment({"text":"hello","target_lang":"en"})[0]["glossary"] == {}


def test_index_bank_identity_is_checked(tmp_path):
    paths = bank(tmp_path, [term("1","eigenvalue","特征值")])
    paths[0].write_text(json.dumps(term("1","eigenvalue","错误术语")), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        TerminologyRAG(*paths, profile="local_terms")


def test_api_model_dimension_and_http_failure_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-private-key")
    client = OpenAIEmbeddings()
    responses = [{"model":"other-model","data":[],"usage":{"total_tokens":1}},
                 {"model":EMBEDDING_MODEL,"data":[{"index":0,"embedding":[1.0]}],"usage":{"total_tokens":1}}]
    for data in responses:
        monkeypatch.setattr(client.client,"post",lambda *a, data=data, **k: httpx.Response(200,json=data))
        with pytest.raises(RuntimeError):
            client.embed(["test"],use_cache=False)
    monkeypatch.setattr(client.client,"post",lambda *a,**k:httpx.Response(401,json={"error":"unit-test-private-key"}))
    with pytest.raises(RuntimeError) as error:
        client.embed(["test"],use_cache=False)
    assert "unit-test-private-key" not in str(error.value)
    client.close()
