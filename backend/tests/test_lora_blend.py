import json
import math
import numpy as np
import pytest
from backend.merge_finetuned_models import blend_lora_adapters, save_safetensors_fp16, load_safetensors_numpy

@pytest.mark.parametrize("rslora", [True, False])
def test_blend_preserves_weighted_updates(tmp_path, rslora):
    dirs = [tmp_path / name for name in ("coding", "phys", "merged")]
    akey = "model.layers.0.self_attn.q_proj.lora_A.weight"
    bkey = akey.replace("lora_A", "lora_B")
    ac = np.array([[1., 2.], [3., 4.]])
    bc = np.array([[2., 1.], [0., 3.]])
    ap = np.array([[4., 0.], [1., 2.]])
    bp = np.array([[1., 3.], [2., 0.]])
    for folder, a, b in zip(dirs[:2], (ac, ap), (bc, bp)):
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": rslora, "base_model_name_or_path": "same-base"}))
        save_safetensors_fp16({akey: a, bkey: b}, folder / "adapter_model.safetensors")
    blend_lora_adapters(*dirs, alpha_coding=.35)
    cfg = json.loads((dirs[2] / "adapter_config.json").read_text())
    merged = load_safetensors_numpy(dirs[2] / "adapter_model.safetensors")
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if rslora else cfg["r"])
    original_scale = 4 / (math.sqrt(2) if rslora else 2)
    np.testing.assert_allclose(scale * (merged[bkey] @ merged[akey]), original_scale * (.35 * bc @ ac + .65 * bp @ ap), rtol=.001)


@pytest.mark.parametrize("ggml_type, elem_size, dt", [
    (2, 32, np.dtype([("d", "<f2"), ("qs", "u1", (16,))])),
    (3, 32, np.dtype([("d", "<f2"), ("m", "<f2"), ("qs", "u1", (16,))])),
    (8, 32, np.dtype([("d", "<f2"), ("qs", "i1", (32,))])),
    (12, 256, np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qs", "u1", (128,))])),
    (13, 256, np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qh", "u1", (32,)), ("qs", "u1", (128,))])),
    (14, 256, np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)), ("scales", "i1", (16,)), ("d", "<f2")])),
])
def test_gguf_zero_delta_roundtrip_quant_types(ggml_type, elem_size, dt):
    from backend.merge_finetuned_models import patch_gguf_tensor_bytes
    n_blocks = 4
    blocks = np.zeros(n_blocks, dtype=dt)
    if "d" in dt.names:
        blocks["d"] = [0.125, -0.25, 0.5, -1.0]
    if "dmin" in dt.names:
        blocks["dmin"] = [0.05, 0.1, -0.2, 0.4]
    if "m" in dt.names:
        blocks["m"] = [-1.0, 0.5, 2.0, -3.0]
    if "scales" in dt.names:
        blocks["scales"] = np.random.randint(1, 64, (n_blocks, dt.fields["scales"][0].shape[0]), dtype=dt.fields["scales"][0].base)
    if "qs" in dt.names:
        blocks["qs"] = np.random.randint(0, 128, (n_blocks, dt.fields["qs"][0].shape[0]), dtype=dt.fields["qs"][0].base)
    if "ql" in dt.names:
        blocks["ql"] = np.random.randint(0, 256, (n_blocks, dt.fields["ql"][0].shape[0]), dtype=dt.fields["ql"][0].base)
    if "qh" in dt.names:
        blocks["qh"] = np.random.randint(0, 256, (n_blocks, dt.fields["qh"][0].shape[0]), dtype=dt.fields["qh"][0].base)

    raw = blocks.tobytes()
    delta = np.zeros((n_blocks, elem_size), dtype=np.float32)
    patched = patch_gguf_tensor_bytes(raw, ggml_type, delta)
    assert raw == patched, f"Mismatch in ggml_type {ggml_type} round-trip with zero delta!"


def test_gguf_zero_delta_roundtrip_float_types():
    from backend.merge_finetuned_models import patch_gguf_tensor_bytes
    # F32
    f32_raw = np.array([1.23, -4.56, 7.89], dtype="<f4").tobytes()
    assert patch_gguf_tensor_bytes(f32_raw, 0, np.zeros(3, dtype=np.float32)) == f32_raw
    # F16
    f16_raw = np.array([1.23, -4.56, 7.89], dtype="<f2").tobytes()
    assert patch_gguf_tensor_bytes(f16_raw, 1, np.zeros(3, dtype=np.float32)) == f16_raw
    # BF16
    bf16_raw = np.array([0x3f80, 0xbf80, 0x4000], dtype="<u2").tobytes()
    assert patch_gguf_tensor_bytes(bf16_raw, 30, np.zeros(3, dtype=np.float32)) == bf16_raw

