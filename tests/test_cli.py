from types import SimpleNamespace

import pytest

from gui_recover.cli import main
from gui_recover.policies import MODEL_REVISION


def test_reused_backend_identity_is_checked_before_browser_start(tmp_path):
    backend = SimpleNamespace(config={"model": "wrong-model"})
    with pytest.raises(ValueError, match="different model"):
        main(["run", "--policy", "showui", "--model", "expected-model", "--out", str(tmp_path / "run")], backend=backend)


def test_reused_adapter_is_checked_before_browser_start(tmp_path):
    backend = SimpleNamespace(config={"model": "expected-model", "revision": MODEL_REVISION,
                                      "prompt_profile": "baseline"})
    with pytest.raises(ValueError, match="different adapter"):
        main(["run", "--policy", "showui", "--model", "expected-model", "--adapter", "trained-adapter",
              "--out", str(tmp_path / "run")], backend=backend)
