from unittest.mock import Mock

import pytest
import requests

from backend import model_runtime
from backend import merge_finetuned_models as merger


def test_cached_startup_never_merges_or_registers_again(tmp_path, monkeypatch):
    gguf = tmp_path / "merged.gguf"
    gguf.write_bytes(b"GGUF")
    (tmp_path / "merge_manifest.json").write_text('{"total_tensors": 410}', encoding="utf-8")
    monkeypatch.setattr(merger, "PATCHED_GGUF_PATH", gguf)
    monkeypatch.setattr(merger, "MERGED_ADAPTER_DIR", tmp_path)
    locate = Mock(side_effect=AssertionError("Cached startup must not rebuild"))
    run = Mock(side_effect=AssertionError("Cached startup must not call ollama create"))
    monkeypatch.setattr(merger, "locate_coding_and_physchem_adapters", locate)
    monkeypatch.setattr(merger.subprocess, "run", run)

    for _ in range(2):
        result = merger.register_unified_model_in_ollama()
        assert result["mode"] == "cached_in_place_gguf"
        assert result["active_model"] == merger.TARGET_OLLAMA_MODEL
    locate.assert_not_called()
    run.assert_not_called()


def test_startup_preloads_finetuned_tag_with_persistent_runner(monkeypatch):
    monkeypatch.setattr(model_runtime.settings, "MODEL_NAME", "base-model")
    monkeypatch.setattr(model_runtime.settings, "OLLAMA_KEEP_ALIVE", -1)
    register = Mock(return_value={"status": "success", "active_model": merger.TARGET_OLLAMA_MODEL})
    monkeypatch.setattr(model_runtime, "register_unified_model_in_ollama", register)
    response = Mock()
    response.json.return_value = {"done": True}
    post = Mock(return_value=response)
    monkeypatch.setattr(model_runtime.requests, "post", post)

    model_runtime.initialize_finetuned_model()

    assert model_runtime.settings.MODEL_NAME == merger.TARGET_OLLAMA_MODEL
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == merger.TARGET_OLLAMA_MODEL
    assert payload["messages"] == []
    assert payload["keep_alive"] == -1
    assert payload["options"] == {
        "num_ctx": model_runtime.settings.NUM_CTX,
        "num_gpu": model_runtime.settings.NUM_GPU,
    }
    response.raise_for_status.assert_called_once()


def test_startup_does_not_silently_fall_back_to_base(monkeypatch):
    monkeypatch.setattr(model_runtime, "register_unified_model_in_ollama", Mock(return_value={"status": "skipped"}))
    post = Mock()
    monkeypatch.setattr(model_runtime.requests, "post", post)
    with pytest.raises(RuntimeError, match="initialization failed"):
        model_runtime.initialize_finetuned_model()
    post.assert_not_called()


def test_startup_fails_if_registered_model_cannot_load(monkeypatch):
    monkeypatch.setattr(model_runtime, "register_unified_model_in_ollama", Mock(return_value={
        "status": "success", "active_model": merger.TARGET_OLLAMA_MODEL,
    }))
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("Model not found")
    monkeypatch.setattr(model_runtime.requests, "post", Mock(return_value=response))
    with pytest.raises(requests.HTTPError, match="Model not found"):
        model_runtime.initialize_finetuned_model()
