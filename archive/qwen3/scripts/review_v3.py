"""Hash-bound Codex review of outputs already individually inspected."""
import hashlib
from pathlib import Path
from collections import Counter

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

SHEETS = {
    "start-dev": ("7aba89f6de494c2fcde52c744d5ae447111adbf32113b6734f67efa9ce48f824", {
        "004-zh-CN": ("minor", "stew泛化为汤，炖菜类别未保留；主要盛菜请求仍在。"),
        "005-zh-CN": ("major", "蛋糕糖衣glaze误译成陶瓷釉料，食品对象词义错误。"),
        "007-en": ("minor", "速率用泛指rate，没有明确speed物理量。"),
    }),
}
SHEETS["selected-dev"] = ("d3b314359240a4bbf091d0d24941252b69442f6fe442b16415ec83e3ec3d2887", SHEETS["start-dev"][1])
SHEETS["cpo-dev"] = ("e672e68b70aeb42c59e6ec9ca513b915c0749a9460107f1b4582d115ce199b3d", SHEETS["start-dev"][1])
SHEETS["start-test"] = ("7d04ecffd0088e5b7d4cbde2609578171bbc2836d16ee6cc4a85ba57c69932cb", {
    "004-zh-CN": ("major", "ramekin小陶瓷盅误译成烤盘，容器对象改变。"),
    "005-en": ("minor", "配料译为additives，容易理解成食品添加剂而非浇头；面条含蛋的主要信息保留。"),
    "007-zh-CN": ("minor", "每个项单独表述不自然、缺少谓语，但总和与单项的范围区别仍可理解。"),
    "008-zh-CN": ("major", "已给河流水流语境，current仍译成电流，忽略消歧语境。"),
})
SHEETS["selected-test"] = ("d340d0aec1df700a500da23575b623cca45509b292c79941baa1363cc429d6a2", SHEETS["start-test"][1])


def persist(name):
    digest, issues = SHEETS[name]
    path = Path(f"runs/v3-{name}.jsonl")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "输出改变，必须重新阅读"
    rows = read_jsonl(path)
    prefix = "v3-dev-" if name.endswith("dev") else "v3-test-"
    decisions = []
    for row in rows:
        key = row["id"].removeprefix(prefix)
        verdict, note = issues.get(key, ("pass", "逐条核对当前原文与语境，接受等义不同措辞；未发现影响含义的问题。"))
        decisions.append({"id": row["id"], "reviewer": "Codex", "at": now(), "verdict": verdict, "note": note,
            "format_valid": "prediction" in row, "output_hash": fingerprint({k: row[k] for k in ("input", "raw", "reference")})})
    assert set(issues).issubset({r["id"].removeprefix(prefix) for r in rows})
    summary = {"at": now(), "count": len(rows), "verdicts": dict(Counter(r["verdict"] for r in decisions)),
        "format_valid": sum(r["format_valid"] for r in decisions), "raw_sha256": digest}
    write_jsonl(f"runs/v3-{name}-semantic.jsonl", decisions)
    write_json(f"runs/v3-{name}-semantic.summary.json", summary)
    return summary


def main():
    print({name: persist(name) for name in SHEETS})


if __name__ == "__main__":
    main()
