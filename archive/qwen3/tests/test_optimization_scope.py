import sys
import pytest
from archive.qwen3.scripts.train_v12_factorial import main


def test_revoked_base_branch_cannot_restart(monkeypatch):
    monkeypatch.setattr(sys,'argv',['train','--arm','base-old'])
    with pytest.raises(ValueError,match='User revoked further base branches'):
        main()
