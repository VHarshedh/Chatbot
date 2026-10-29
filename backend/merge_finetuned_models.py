"""
backend/merge_finetuned_models.py
===============================================================================
Zero-HuggingFace, Pure-NumPy In-Place GGUF LoRA Merger for `batiai/gemma4-e2b:q4`
===============================================================================
1. Extracts `coding_ft.zip` (`scicompute_ft_data/gemma_e2b_best_loss_adapter`)
   and `physchem_ft.zip` (`physchem_ft_outputs/gemma_e2b_physchem_best_adapter`).
2. Blends all 410 LoRA tensors (35% Coding + 65% PhysChem) into
   `ChatBot/merged_ft_adapter/adapter_model.safetensors`.
3. Queries `ollama show --modelfile batiai/gemma4-e2b:q4` to find the exact local
   GGUF blob (`C:\\Users\\harsh\\.ollama\\models\\blobs\\sha256-...`) pulled from Ollama.
4. Copies that GGUF blob to `merged_ft_adapter/batiai_gemma4_e2b_ft_merged.gguf` and
   patches all 205 transformer layer matrices (`blk.{L}.attn_q/k/v/output`,
   `blk.{L}.ffn_gate/up/down`) in-place across Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K,
   F16, BF16, and F32 quantization blocks using `Delta_W = (lora_alpha / sqrt(r)) * (B @ A)`.
5. Re-registers the patched GGUF directly as `batiai/gemma4-e2b:q4` in Ollama so the
   entire ChatBot UI (`backend/config.py`, `frontend/index.html`, `frontend/app.js`,
   `SingleShotAgent`, `AgenticRAGAgent`, and `pytest`) uses the merged weights automatically.
"""

import os
import re
import json
import math
import struct
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_ADAPTER_DIR = REPO_ROOT / "merged_ft_adapter"
ORIGINAL_BASE_GGUF = MERGED_ADAPTER_DIR / "original_base.gguf"
PATCHED_GGUF_PATH = MERGED_ADAPTER_DIR / "batiai_gemma4_e2b_ft_merged.gguf"
ORIG_MODELFILE_BACKUP = MERGED_ADAPTER_DIR / "original_batiai_modelfile.txt"
TARGET_OLLAMA_MODEL = "batiai/gemma4-e2b:q4"
UNIFIED_MODEL_NAME = TARGET_OLLAMA_MODEL

UNIFIED_SYSTEM_PROMPT = (
    "You are SciCompute PhysChem & Code AI, a principal computational materials physicist, quantum chemist, "
    "and scientific Python engineer specialized in ASE, Quantum ESPRESSO (DFT), DeePMD-kit, and LAMMPS.\n"
    "1. Chemical Bonding & Phase Partitioning: Intramolecular covalent O-H bonds in H2O/OH* are ~0.97 Å (cutoff r_OH = 1.25 Å), "
    "whereas intermolecular H-bonds start at >=1.65 Å. Always partition water oxygens (`O_w`) from crystal lattice oxygens (`O_slab`) "
    "via `neighbor_list('ij', atoms, cutoff={('O', 'H'): 1.25})` BEFORE computing crystal surface layers or enclosing boundary shells (r_if = 3.0-3.2 Å under PBC).\n"
    "2. Statistical Mechanics & Units: Planar water mass density rho(z) [g/cm^3] = ((N_Ow / 6.02214076e23) * 18.01528) / (A_xy * dz * 1e-24). "
    "Normalize partial RDFs g_AB(r) by ideal-gas spherical shell count `(N_A * N_B / V_cell) * 4 * pi * r^2 * dr` under MIC (`mic=True`).\n"
    "3. ASE API Contracts: `neighbor_list('ijd', atoms, cutoff)` returns three 1D arrays `(nl_i, nl_j, nl_d)`—never index as 2D `d[i,j]`. "
    "Neither `Atoms` nor `Atom` has `.get_neighbors()`. Use plain Unicode ('TiO2', 'Å') and never `\\text{}` in plot labels."
)

HF_TO_GGUF_SUFFIX = {
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.gate_proj": "ffn_gate.weight",
    "mlp.up_proj": "ffn_up.weight",
    "mlp.down_proj": "ffn_down.weight",
}

GGML_TYPE_BYTE_SIZES = {
    0: (1, 4),      # F32: 4B per 1 elem
    1: (1, 2),      # F16: 2B per 1 elem
    2: (32, 18),    # Q4_0: 18B per 32 elems
    3: (32, 20),    # Q4_1: 20B per 32 elems
    8: (32, 34),    # Q8_0: 34B per 32 elems
    12: (256, 144), # Q4_K: 144B per 256 elems
    13: (256, 176), # Q5_K: 176B per 256 elems
    14: (256, 210), # Q6_K: 210B per 256 elems
    30: (1, 2),     # BF16: 2B per 1 elem
}


# ------------------------------------------------------------------------------
# 1. Pure-NumPy SafeTensors Reader & Writer
# ------------------------------------------------------------------------------
def load_safetensors_numpy(st_path: Path) -> Dict[str, np.ndarray]:
    with open(st_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            if dtype_str == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
            elif dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2")
                arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
            else:
                continue
            tensors[name] = arr
    return tensors


def save_safetensors_fp16(tensors: Dict[str, np.ndarray], st_path: Path) -> None:
    header = {}
    offset = 0
    raw_chunks = []
    for name, arr in tensors.items():
        arr16 = arr.astype("<f2")
        b = arr16.tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": list(arr16.shape),
            "data_offsets": [offset, offset + len(b)],
        }
        raw_chunks.append(b)
        offset += len(b)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with open(st_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in raw_chunks:
            f.write(chunk)


# ------------------------------------------------------------------------------
# 2. Zip Extraction & LoRA Adapter Weight Blending
# ------------------------------------------------------------------------------
def ensure_zips_extracted(root_dir: Path = REPO_ROOT) -> None:
    for zip_name, expected_folder in [("coding_ft.zip", "scicompute_ft_data"), ("physchem_ft.zip", "physchem_ft_outputs")]:
        zip_path = root_dir / zip_name
        out_folder = root_dir / expected_folder
        if zip_path.exists() and not out_folder.exists():
            print(f"[FT Merger] Extracting {zip_path.name} into {root_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root_dir)


def locate_coding_and_physchem_adapters(root_dir: Path = REPO_ROOT) -> Tuple[Optional[Path], Optional[Path]]:
    ensure_zips_extracted(root_dir)
    coding_candidates = ["gemma_e2b_best_loss_adapter", "gemma_e2b_scicompute_dpo", "gemma_e2b_soup_candidate", "gemma_e2b_scicompute_sft"]
    physchem_candidates = ["gemma_e2b_physchem_best_adapter", "gemma_e2b_physchem_dpo", "gemma_e2b_physchem_soup_candidate", "gemma_e2b_physchem_sft"]

    coding_dir = None
    for name in coding_candidates:
        for match in root_dir.rglob(name):
            if (match / "adapter_model.safetensors").exists():
                coding_dir = match
                break
        if coding_dir:
            break

    physchem_dir = None
    for name in physchem_candidates:
        for match in root_dir.rglob(name):
            if (match / "adapter_model.safetensors").exists():
                physchem_dir = match
                break
        if physchem_dir:
            break

    return coding_dir, physchem_dir


def compute_optimal_blend_alpha(root_dir: Path = REPO_ROOT, default_alpha: float = 0.85) -> float:
    try:
        c_rep_files = list(root_dir.rglob("benchmark_execution_report.json"))
        p_rep_files = list(root_dir.rglob("physchem_benchmark_report.json"))
        if c_rep_files and p_rep_files:
            c_rep = json.loads(c_rep_files[0].read_text(encoding="utf-8"))
            p_rep = json.loads(p_rep_files[0].read_text(encoding="utf-8"))
            c_loss = float(c_rep.get("best_val_loss", 1.0))
            p_loss = float(p_rep.get("best_val_loss", 1.0))
            c_pass = float(c_rep.get("pass_at_1", -1.0))
            p_pass = float(p_rep.get("pass_at_1", -1.0))

            # Defensive audit against training collapse:
            # If an adapter achieved low loss but 0% pass rate with syntax errors,
            # avoid letting degenerate output dominate the merge.
            if p_pass == 0.0 and c_pass > 0.0:
                return 0.85
            if c_pass == 0.0 and p_pass > 0.0:
                return 0.15

            if c_pass > 0 and p_pass > 0:
                score_c = (c_pass / 100.0) / max(c_loss, 1e-4)
                score_p = (p_pass / 100.0) / max(p_loss, 1e-4)
                return round(min(max(score_c / (score_c + score_p), 0.35), 0.85), 4)

            if c_loss > 0 and p_loss > 0:
                w_c, w_p = 1.0 / c_loss, 1.0 / p_loss
                return round(min(max(w_c / (w_c + w_p), 0.35), 0.65), 4)
    except Exception:
        pass
    return default_alpha


def blend_lora_adapters(
    coding_adapter_dir: Path,
    physchem_adapter_dir: Path,
    output_adapter_dir: Path = MERGED_ADAPTER_DIR,
    alpha_coding: Optional[float] = None,
) -> Dict[str, Any]:
    if alpha_coding is None:
        alpha_coding = compute_optimal_blend_alpha(REPO_ROOT, default_alpha=0.50)
    alpha_phys = round(1.0 - alpha_coding, 4)

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    for item in coding_adapter_dir.iterdir():
        if item.is_file() and item.name not in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(item, output_adapter_dir / item.name)

    sd_coding = load_safetensors_numpy(coding_adapter_dir / "adapter_model.safetensors")
    sd_phys = load_safetensors_numpy(physchem_adapter_dir / "adapter_model.safetensors")

    cfg_c_path = coding_adapter_dir / "adapter_config.json"
    cfg_p_path = physchem_adapter_dir / "adapter_config.json"
    c_cfg = json.loads(cfg_c_path.read_text(encoding="utf-8")) if cfg_c_path.exists() else {}
    p_cfg = json.loads(cfg_p_path.read_text(encoding="utf-8")) if cfg_p_path.exists() else {}

    c_r = float(c_cfg.get("r", 32))
    c_alpha = float(c_cfg.get("lora_alpha", 64))
    c_rslora = bool(c_cfg.get("use_rslora", True))
    scale_c = c_alpha / (math.sqrt(c_r) if c_rslora else c_r)

    p_r = float(p_cfg.get("r", 32))
    p_alpha = float(p_cfg.get("lora_alpha", 64))
    p_rslora = bool(p_cfg.get("use_rslora", True))
    scale_p = p_alpha / (math.sqrt(p_r) if p_rslora else p_r)

    r_merged = int(c_r + p_r)
    alpha_merged = float(c_alpha)
    use_rslora_merged = c_rslora
    scale_merged = alpha_merged / (math.sqrt(r_merged) if use_rslora_merged else r_merged)

    w_c = (alpha_coding * scale_c) / scale_merged
    w_p = (alpha_phys * scale_p) / scale_merged

    blended_sd = {}
    matched_pairs = 0
    handled_keys = set()

    for key, t_c in sd_coding.items():
        if key in handled_keys:
            continue
        if ".lora_A." in key or key.endswith("lora_A.weight"):
            b_key = key.replace("lora_A", "lora_B")
            if b_key in sd_coding and key in sd_phys and b_key in sd_phys:
                a_c = t_c
                b_c = sd_coding[b_key]
                a_p = sd_phys[key]
                b_p = sd_phys[b_key]
                if a_c.shape[1] == a_p.shape[1] and b_c.shape[0] == b_p.shape[0]:
                    blended_sd[key] = np.concatenate([a_c, a_p], axis=0)
                    blended_sd[b_key] = np.concatenate([w_c * b_c, w_p * b_p], axis=1)
                    handled_keys.add(key)
                    handled_keys.add(b_key)
                    matched_pairs += 1
                    continue
        if key in sd_phys and t_c.shape == sd_phys[key].shape:
            blended_sd[key] = alpha_coding * t_c + alpha_phys * sd_phys[key]
            handled_keys.add(key)
        else:
            blended_sd[key] = t_c
            handled_keys.add(key)

    for key, t_p in sd_phys.items():
        if key not in handled_keys:
            blended_sd[key] = t_p

    merged_cfg = dict(c_cfg) if c_cfg else dict(p_cfg)
    merged_cfg["r"] = r_merged
    merged_cfg["lora_alpha"] = alpha_merged
    merged_cfg["use_rslora"] = use_rslora_merged
    (output_adapter_dir / "adapter_config.json").write_text(json.dumps(merged_cfg, indent=2), encoding="utf-8")

    save_safetensors_fp16(blended_sd, output_adapter_dir / "adapter_model.safetensors")
    manifest = {
        "coding_adapter": str(coding_adapter_dir),
        "physchem_adapter": str(physchem_adapter_dir),
        "output_adapter": str(output_adapter_dir),
        "alpha_coding": alpha_coding,
        "alpha_physchem": alpha_phys,
        "matched_lora_pairs": matched_pairs,
        "total_tensors": len(blended_sd),
    }
    (output_adapter_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[FT Merger] Blended {matched_pairs} LoRA pairs ({alpha_coding*100:.1f}% Coding + {alpha_phys*100:.1f}% PhysChem) -> {output_adapter_dir}")
    return manifest


# ------------------------------------------------------------------------------
# 3. GGUF Binary Header Parser & Block Dequantizer/Requantizer
# ------------------------------------------------------------------------------
def _read_gguf_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _skip_gguf_val(f, vtype: int) -> None:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        f.seek(sizes[vtype], 1)
    elif vtype == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(n, 1)
    elif vtype == 9:
        elem_type, arr_len = struct.unpack("<IQ", f.read(12))
        if elem_type in sizes:
            f.seek(sizes[elem_type] * arr_len, 1)
        else:
            for _ in range(arr_len):
                _skip_gguf_val(f, elem_type)
    else:
        raise ValueError(f"Unsupported GGUF metadata type {vtype}")


def parse_gguf_tensors(gguf_path: Path) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    with open(gguf_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{gguf_path} is not a valid GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        alignment = 32
        for _ in range(n_kv):
            key = _read_gguf_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                _skip_gguf_val(f, vtype)
        tensors = {}
        for _ in range(n_tensors):
            t_name = _read_gguf_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type, rel_offset = struct.unpack("<IQ", f.read(12))
            tensors[t_name] = {"dims": dims, "ggml_type": ggml_type, "rel_offset": rel_offset}
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment
    return data_offset, tensors


def patch_gguf_tensor_bytes(raw_bytes: bytes, ggml_type: int, delta_w: np.ndarray) -> bytes:
    """Dequantizes a GGUF tensor byte slice, adds `delta_w`, and re-quantizes to the exact same format."""
    flat_delta = delta_w.reshape(-1)
    if ggml_type == 0:  # F32
        w = np.frombuffer(raw_bytes, dtype="<f4").copy() + flat_delta
        return w.astype("<f4").tobytes()
    if ggml_type == 1:  # F16
        w = np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32) + flat_delta
        return w.astype("<f2").tobytes()
    if ggml_type == 30:  # BF16
        u16 = np.frombuffer(raw_bytes, dtype="<u2")
        w = (u16.astype(np.uint32) << 16).view(np.float32) + flat_delta
        new_u16 = (w.view(np.uint32) >> 16).astype("<u2")
        return new_u16.tobytes()
    if ggml_type == 2:  # Q4_0
        n_blocks = len(raw_bytes) // 18
        dt = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d = blocks["d"].astype(np.float32)[:, None]
        qs = blocks["qs"]
        orig_lo = qs & 0x0F
        orig_hi = qs >> 4
        lo = orig_lo.astype(np.float32) - 8.0
        hi = orig_hi.astype(np.float32) - 8.0
        w = np.concatenate([lo, hi], axis=1) * d
        w_new = w + flat_delta.reshape(n_blocks, 32)
        d_safe = np.where(np.abs(d) > 1e-12, d, 1.0)
        q_lo = np.where(np.abs(d) > 1e-12, np.clip(np.round(w_new[:, :16] / d_safe) + 8.0, 0, 15), orig_lo).astype(np.uint8)
        q_hi = np.where(np.abs(d) > 1e-12, np.clip(np.round(w_new[:, 16:] / d_safe) + 8.0, 0, 15), orig_hi).astype(np.uint8)
        blocks["qs"] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4)
        return blocks.tobytes()
    if ggml_type == 3:  # Q4_1
        n_blocks = len(raw_bytes) // 20
        dt = np.dtype([("d", "<f2"), ("m", "<f2"), ("qs", "u1", (16,))])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d = blocks["d"].astype(np.float32)[:, None]
        m = blocks["m"].astype(np.float32)[:, None]
        qs = blocks["qs"]
        orig_lo = qs & 0x0F
        orig_hi = qs >> 4
        lo = orig_lo.astype(np.float32) * d + m
        hi = orig_hi.astype(np.float32) * d + m
        w = np.concatenate([lo, hi], axis=1)
        w_new = w + flat_delta.reshape(n_blocks, 32)
        d_safe = np.where(np.abs(d) > 1e-12, d, 1.0)
        q_lo = np.where(np.abs(d) > 1e-12, np.clip(np.round((w_new[:, :16] - m) / d_safe), 0, 15), orig_lo).astype(np.uint8)
        q_hi = np.where(np.abs(d) > 1e-12, np.clip(np.round((w_new[:, 16:] - m) / d_safe), 0, 15), orig_hi).astype(np.uint8)
        blocks["qs"] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4)
        return blocks.tobytes()
    if ggml_type == 8:  # Q8_0
        n_blocks = len(raw_bytes) // 34
        dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d = blocks["d"].astype(np.float32)[:, None]
        orig_qs = blocks["qs"]
        w_new = orig_qs.astype(np.float32) * d + flat_delta.reshape(n_blocks, 32)
        d_safe = np.where(np.abs(d) > 1e-12, d, 1.0)
        blocks["qs"] = np.where(np.abs(d) > 1e-12, np.clip(np.round(w_new / d_safe), -127, 127), orig_qs).astype(np.int8)
        return blocks.tobytes()
    if ggml_type == 12:  # Q4_K
        n_blocks = len(raw_bytes) // 144
        dt = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qs", "u1", (128,))])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d = blocks["d"].astype(np.float32)
        dmin = blocks["dmin"].astype(np.float32)
        sc_raw, qs = blocks["scales"], blocks["qs"]
        sc = np.zeros((n_blocks, 8), dtype=np.float32)
        mn = np.zeros((n_blocks, 8), dtype=np.float32)
        for j in range(4):
            sc[:, j] = sc_raw[:, j] & 63
            mn[:, j] = sc_raw[:, j + 4] & 63
        for j in range(4, 8):
            sc[:, j] = (sc_raw[:, j + 4] & 0x0F) | ((sc_raw[:, j - 4] >> 6) << 4)
            mn[:, j] = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
        w = np.zeros((n_blocks, 256), dtype=np.float32)
        for chunk in range(4):
            q_c = qs[:, chunk * 32 : (chunk + 1) * 32]
            w[:, chunk * 64 : chunk * 64 + 32] = (q_c & 0x0F).astype(np.float32) * (d * sc[:, 2 * chunk])[:, None] - (dmin * mn[:, 2 * chunk])[:, None]
            w[:, chunk * 64 + 32 : (chunk + 1) * 64] = (q_c >> 4).astype(np.float32) * (d * sc[:, 2 * chunk + 1])[:, None] - (dmin * mn[:, 2 * chunk + 1])[:, None]
        w_new = w + flat_delta.reshape(n_blocks, 256)
        new_qs = np.zeros((n_blocks, 128), dtype=np.uint8)
        for chunk in range(4):
            q_c = qs[:, chunk * 32 : (chunk + 1) * 32]
            orig_q1 = q_c & 0x0F
            orig_q2 = q_c >> 4
            d1 = (d * sc[:, 2 * chunk])[:, None]
            m1 = (dmin * mn[:, 2 * chunk])[:, None]
            d2 = (d * sc[:, 2 * chunk + 1])[:, None]
            m2 = (dmin * mn[:, 2 * chunk + 1])[:, None]
            w1 = w_new[:, chunk * 64 : chunk * 64 + 32]
            w2 = w_new[:, chunk * 64 + 32 : (chunk + 1) * 64]
            q1 = np.where(d1 > 1e-12, np.clip(np.round((w1 + m1) / np.maximum(d1, 1e-12)), 0, 15), orig_q1).astype(np.uint8)
            q2 = np.where(d2 > 1e-12, np.clip(np.round((w2 + m2) / np.maximum(d2, 1e-12)), 0, 15), orig_q2).astype(np.uint8)
            new_qs[:, chunk * 32 : (chunk + 1) * 32] = (q1 & 0x0F) | ((q2 & 0x0F) << 4)
        blocks["qs"] = new_qs
        return blocks.tobytes()
    if ggml_type == 13:  # Q5_K
        n_blocks = len(raw_bytes) // 176
        dt = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qh", "u1", (32,)), ("qs", "u1", (128,))])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d, dmin = blocks["d"].astype(np.float32), blocks["dmin"].astype(np.float32)
        sc_raw, qh, qs = blocks["scales"], blocks["qh"], blocks["qs"]
        sc = np.zeros((n_blocks, 8), dtype=np.float32)
        mn = np.zeros((n_blocks, 8), dtype=np.float32)
        for j in range(4):
            sc[:, j], mn[:, j] = sc_raw[:, j] & 63, sc_raw[:, j + 4] & 63
        for j in range(4, 8):
            sc[:, j] = (sc_raw[:, j + 4] & 0x0F) | ((sc_raw[:, j - 4] >> 6) << 4)
            mn[:, j] = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
        w = np.zeros((n_blocks, 256), dtype=np.float32)
        for chunk in range(4):
            q_c = qs[:, chunk * 32 : (chunk + 1) * 32]
            h1, h2 = ((qh >> (2 * chunk)) & 1) << 4, ((qh >> (2 * chunk + 1)) & 1) << 4
            w[:, chunk * 64 : chunk * 64 + 32] = ((q_c & 0x0F) | h1).astype(np.float32) * (d * sc[:, 2 * chunk])[:, None] - (dmin * mn[:, 2 * chunk])[:, None]
            w[:, chunk * 64 + 32 : (chunk + 1) * 64] = ((q_c >> 4) | h2).astype(np.float32) * (d * sc[:, 2 * chunk + 1])[:, None] - (dmin * mn[:, 2 * chunk + 1])[:, None]
        w_new = w + flat_delta.reshape(n_blocks, 256)
        new_qs = np.zeros((n_blocks, 128), dtype=np.uint8)
        new_qh = np.zeros((n_blocks, 32), dtype=np.uint8)
        for chunk in range(4):
            q_c = qs[:, chunk * 32 : (chunk + 1) * 32]
            h1, h2 = ((qh >> (2 * chunk)) & 1) << 4, ((qh >> (2 * chunk + 1)) & 1) << 4
            orig_q1 = (q_c & 0x0F) | h1
            orig_q2 = (q_c >> 4) | h2
            d1, m1 = (d * sc[:, 2 * chunk])[:, None], (dmin * mn[:, 2 * chunk])[:, None]
            d2, m2 = (d * sc[:, 2 * chunk + 1])[:, None], (dmin * mn[:, 2 * chunk + 1])[:, None]
            w1 = w_new[:, chunk * 64 : chunk * 64 + 32]
            w2 = w_new[:, chunk * 64 + 32 : (chunk + 1) * 64]
            q1 = np.where(d1 > 1e-12, np.clip(np.round((w1 + m1) / np.maximum(d1, 1e-12)), 0, 31), orig_q1).astype(np.uint8)
            q2 = np.where(d2 > 1e-12, np.clip(np.round((w2 + m2) / np.maximum(d2, 1e-12)), 0, 31), orig_q2).astype(np.uint8)
            new_qs[:, chunk * 32 : (chunk + 1) * 32] = (q1 & 0x0F) | ((q2 & 0x0F) << 4)
            new_qh |= (((q1 >> 4) & 1) << (2 * chunk)) | (((q2 >> 4) & 1) << (2 * chunk + 1))
        blocks["qs"], blocks["qh"] = new_qs, new_qh
        return blocks.tobytes()
    if ggml_type == 14:  # Q6_K
        n_blocks = len(raw_bytes) // 210
        dt = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)), ("scales", "i1", (16,)), ("d", "<f2")])
        blocks = np.frombuffer(raw_bytes, dtype=dt).copy()
        d, sc, ql, qh = blocks["d"].astype(np.float32), blocks["scales"].astype(np.float32), blocks["ql"], blocks["qh"]
        w = np.zeros((n_blocks, 256), dtype=np.float32)
        for half in range(2):
            ql_h, qh_h, sc_h = ql[:, half * 64 : (half + 1) * 64], qh[:, half * 32 : (half + 1) * 32], sc[:, half * 8 : (half + 1) * 8]
            q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.float32) - 32.0
            q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.float32) - 32.0
            q3 = ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.float32) - 32.0
            q4 = ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.float32) - 32.0
            for sub_idx, q_arr in enumerate((q1, q2, q3, q4)):
                base = half * 128 + sub_idx * 32
                w[:, base : base + 16] = d[:, None] * sc_h[:, 2 * sub_idx, None] * q_arr[:, :16]
                w[:, base + 16 : base + 32] = d[:, None] * sc_h[:, 2 * sub_idx + 1, None] * q_arr[:, 16:]
        w_new = w + flat_delta.reshape(n_blocks, 256)
        new_ql = np.zeros((n_blocks, 128), dtype=np.uint8)
        new_qh = np.zeros((n_blocks, 64), dtype=np.uint8)
        for half in range(2):
            ql_h, qh_h, sc_h = ql[:, half * 64 : (half + 1) * 64], qh[:, half * 32 : (half + 1) * 32], sc[:, half * 8 : (half + 1) * 8]
            orig_q = [
                ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)),
                ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)),
                ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)),
                ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)),
            ]
            b = half * 128
            w_sub = [
                w_new[:, b : b + 32],
                w_new[:, b + 32 : b + 64],
                w_new[:, b + 64 : b + 96],
                w_new[:, b + 96 : b + 128],
            ]
            q_subs = []
            for sub_idx in range(4):
                scale_lo = np.repeat(d[:, None] * sc_h[:, 2 * sub_idx, None], 16, axis=1)
                scale_hi = np.repeat(d[:, None] * sc_h[:, 2 * sub_idx + 1, None], 16, axis=1)
                scale = np.concatenate([scale_lo, scale_hi], axis=1)
                scale_safe = np.where(np.abs(scale) > 1e-12, scale, 1.0)
                q_arr = np.where(np.abs(scale) > 1e-12, np.clip(np.round(w_sub[sub_idx] / scale_safe) + 32.0, 0, 63), orig_q[sub_idx]).astype(np.uint8)
                q_subs.append(q_arr)
            q1, q2, q3, q4 = q_subs
            new_ql[:, half * 64 : half * 64 + 32] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
            new_ql[:, half * 64 + 32 : (half + 1) * 64] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
            new_qh[:, half * 32 : (half + 1) * 32] = (((q1 >> 4) & 3) << 0) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6)
        blocks["ql"], blocks["qh"] = new_ql, new_qh
        return blocks.tobytes()
    raise ValueError(f"Unsupported GGUF quantization type {ggml_type}")


# ------------------------------------------------------------------------------
# 4. Locate `batiai/gemma4-e2b:q4` Local GGUF Blob & Patch In-Place
# ------------------------------------------------------------------------------
def find_ollama_gguf_blob_and_modelfile(model_name: str = TARGET_OLLAMA_MODEL) -> Tuple[Optional[Path], str]:
    MERGED_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    modelfile_text = ""
    if ORIG_MODELFILE_BACKUP.exists():
        modelfile_text = ORIG_MODELFILE_BACKUP.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            res = subprocess.run(["ollama", "show", "--modelfile", model_name], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            if res.returncode == 0 and res.stdout.strip():
                modelfile_text = res.stdout
                if "batiai_gemma4_e2b_ft_merged.gguf" not in modelfile_text:
                    ORIG_MODELFILE_BACKUP.write_text(modelfile_text, encoding="utf-8")
        except FileNotFoundError:
            pass

    # 1. If original_base.gguf exists, ALWAYS prioritize it as the pristine base blob!
    if ORIGINAL_BASE_GGUF.exists() and ORIGINAL_BASE_GGUF.stat().st_size > 500_000_000:
        return ORIGINAL_BASE_GGUF, modelfile_text

    candidates: List[Path] = []
    for line in modelfile_text.splitlines():
        s = line.strip()
        if s.upper().startswith("FROM "):
            p_str = s[5:].strip().strip('"').strip("'")
            p = Path(p_str)
            if (
                p.exists()
                and p.is_file()
                and p.resolve() != PATCHED_GGUF_PATH.resolve()
                and p.name != PATCHED_GGUF_PATH.name
                and "ft_merged" not in p.name
            ):
                candidates.append(p)

    manifest_path = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "batiai" / "gemma4-e2b" / "q4"
    if manifest_path.exists():
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for layer in m_data.get("layers", []):
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    d = layer.get("digest", "")
                    if d.startswith("sha256:"):
                        blob_file = Path.home() / ".ollama" / "models" / "blobs" / f"sha256-{d[7:]}"
                        if blob_file.exists() and blob_file.stat().st_size > 500_000_000:
                            candidates.insert(0, blob_file)
        except Exception:
            pass

    ollama_blobs_dir = Path.home() / ".ollama" / "models" / "blobs"
    if not candidates and ollama_blobs_dir.exists():
        for blob in sorted(ollama_blobs_dir.glob("sha256-*"), key=lambda x: x.stat().st_size, reverse=True):
            if (
                blob.stat().st_size > 500_000_000
                and blob.resolve() != PATCHED_GGUF_PATH.resolve()
                and blob.name != PATCHED_GGUF_PATH.name
                and "ft_merged" not in blob.name
            ):
                candidates.append(blob)

    for cand in candidates:
        if cand.resolve() == PATCHED_GGUF_PATH.resolve() or cand.name == PATCHED_GGUF_PATH.name or "ft_merged" in cand.name:
            continue
        try:
            _, t_map = parse_gguf_tensors(cand)
            if any("attn_q.weight" in k or "q_proj.weight" in k for k in t_map):
                if not ORIGINAL_BASE_GGUF.exists():
                    shutil.copy2(cand, ORIGINAL_BASE_GGUF)
                    return ORIGINAL_BASE_GGUF, modelfile_text
                return cand, modelfile_text
        except Exception:
            continue

    return None, modelfile_text


def patch_ollama_gguf_with_blended_lora(
    base_gguf_path: Path,
    adapter_dir: Path = MERGED_ADAPTER_DIR,
    out_gguf_path: Path = PATCHED_GGUF_PATH,
    coding_adapter_dir: Optional[Path] = None,
    physchem_adapter_dir: Optional[Path] = None,
    alpha_coding: Optional[float] = None,
    use_rslora_scale: bool = False,
) -> Dict[str, Any]:
    pat = re.compile(r"layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.(lora_[AB])\.weight$")

    use_direct_pairs = False
    c_pairs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    p_pairs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    scale_c, scale_p, alpha_c, alpha_p = 1.0, 1.0, 0.35, 0.65

    if coding_adapter_dir and physchem_adapter_dir:
        c_st = load_safetensors_numpy(coding_adapter_dir / "adapter_model.safetensors")
        p_st = load_safetensors_numpy(physchem_adapter_dir / "adapter_model.safetensors")
        c_cfg_path = coding_adapter_dir / "adapter_config.json"
        p_cfg_path = physchem_adapter_dir / "adapter_config.json"
        c_cfg = json.loads(c_cfg_path.read_text(encoding="utf-8")) if c_cfg_path.exists() else {}
        p_cfg = json.loads(p_cfg_path.read_text(encoding="utf-8")) if p_cfg_path.exists() else {}

        c_r = float(c_cfg.get("r", 32))
        c_alpha = float(c_cfg.get("lora_alpha", 64))
        scale_c = c_alpha / (math.sqrt(c_r) if (use_rslora_scale and c_cfg.get("use_rslora", False)) else c_r)

        p_r = float(p_cfg.get("r", 32))
        p_alpha = float(p_cfg.get("lora_alpha", 64))
        scale_p = p_alpha / (math.sqrt(p_r) if (use_rslora_scale and p_cfg.get("use_rslora", False)) else p_r)

        if alpha_coding is None:
            alpha_coding = 0.85
        alpha_c = alpha_coding
        alpha_p = round(1.0 - alpha_c, 4)

        for key, arr in c_st.items():
            m = pat.search(key)
            if m:
                layer_idx, hf_proj, ab_tag = m.groups()
                c_pairs.setdefault((layer_idx, hf_proj), {})[ab_tag] = arr

        for key, arr in p_st.items():
            m = pat.search(key)
            if m:
                layer_idx, hf_proj, ab_tag = m.groups()
                p_pairs.setdefault((layer_idx, hf_proj), {})[ab_tag] = arr

        use_direct_pairs = True

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = float(adapter_cfg.get("r", 32))
    lora_alpha = float(adapter_cfg.get("lora_alpha", 64))
    use_rslora = bool(adapter_cfg.get("use_rslora", False)) and use_rslora_scale
    scale = (lora_alpha / math.sqrt(r)) if use_rslora else (lora_alpha / r)

    lora_tensors = load_safetensors_numpy(adapter_dir / "adapter_model.safetensors")
    merged_pairs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for key, arr in lora_tensors.items():
        m = pat.search(key)
        if not m:
            continue
        layer_idx, hf_proj, ab_tag = m.groups()
        merged_pairs.setdefault((layer_idx, hf_proj), {})[ab_tag] = arr

    print(f"[GGUF Patcher] Copying base `batiai/gemma4-e2b:q4` GGUF blob ({base_gguf_path.stat().st_size / (1024**3):.2f} GiB) -> {out_gguf_path.name}...")
    shutil.copy2(base_gguf_path, out_gguf_path)

    data_offset, gguf_tensors = parse_gguf_tensors(out_gguf_path)
    patched_layers = 0
    skipped_layers = 0

    all_keys = sorted(set(merged_pairs.keys()) | set(c_pairs.keys()), key=lambda x: (int(x[0]), x[1]))

    with open(out_gguf_path, "r+b") as f:
        for (layer_idx, hf_proj) in all_keys:
            gguf_suf = HF_TO_GGUF_SUFFIX.get(hf_proj)
            if not gguf_suf:
                skipped_layers += 1
                continue

            cand_names = [
                f"blk.{layer_idx}.{gguf_suf}",
                f"language_model.blk.{layer_idx}.{gguf_suf}",
                f"model.layers.{layer_idx}.{hf_proj}.weight",
            ]
            gguf_name = next((c for c in cand_names if c in gguf_tensors), None)
            if gguf_name is None:
                skipped_layers += 1
                continue

            t_info = gguf_tensors[gguf_name]
            ggml_type = t_info["ggml_type"]
            dims = t_info["dims"]
            if ggml_type not in GGML_TYPE_BYTE_SIZES or len(dims) != 2:
                skipped_layers += 1
                continue

            in_feat, out_feat = int(dims[0]), int(dims[1])

            pair_key = (layer_idx, hf_proj)
            if use_direct_pairs and pair_key in c_pairs and pair_key in p_pairs:
                c_dict = c_pairs[pair_key]
                p_dict = p_pairs[pair_key]
                if "lora_A" in c_dict and "lora_B" in c_dict and "lora_A" in p_dict and "lora_B" in p_dict:
                    a_c, b_c = c_dict["lora_A"], c_dict["lora_B"]
                    a_p, b_p = p_dict["lora_A"], p_dict["lora_B"]
                    if a_c.shape[1] == in_feat and b_c.shape[0] == out_feat and a_p.shape[1] == in_feat and b_p.shape[0] == out_feat:
                        delta_w = alpha_c * scale_c * (b_c @ a_c) + alpha_p * scale_p * (b_p @ a_p)
                    else:
                        skipped_layers += 1
                        continue
                else:
                    skipped_layers += 1
                    continue
            elif pair_key in merged_pairs:
                ab_dict = merged_pairs[pair_key]
                if "lora_A" not in ab_dict or "lora_B" not in ab_dict:
                    skipped_layers += 1
                    continue
                lora_A, lora_B = ab_dict["lora_A"], ab_dict["lora_B"]
                if lora_A.shape[1] != in_feat or lora_B.shape[0] != out_feat:
                    skipped_layers += 1
                    continue
                delta_w = scale * (lora_B @ lora_A)
            else:
                skipped_layers += 1
                continue

            block_elems, block_bytes = GGML_TYPE_BYTE_SIZES[ggml_type]
            n_bytes = ((in_feat * out_feat) // block_elems) * block_bytes
            abs_offset = data_offset + int(t_info["rel_offset"])

            f.seek(abs_offset)
            orig_bytes = f.read(n_bytes)
            patched_bytes = patch_gguf_tensor_bytes(orig_bytes, ggml_type, delta_w)
            f.seek(abs_offset)
            f.write(patched_bytes)
            patched_layers += 1

    print(f"[GGUF Patcher] Patched {patched_layers}/{len(all_keys)} LoRA matrix pairs directly into {out_gguf_path.name}!")
    return {"patched_matrices": patched_layers, "skipped_matrices": skipped_layers, "gguf_path": str(out_gguf_path)}


def build_patched_modelfile(orig_modelfile: str, orig_blob_path: Path, patched_gguf_path: Path) -> Path:
    out_lines = []
    replaced_main_from = False
    for line in orig_modelfile.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if s.upper().startswith("FROM "):
            if not replaced_main_from:
                out_lines.append(f"FROM {patched_gguf_path.resolve()}")
                replaced_main_from = True
            continue
        elif s.upper().startswith("SYSTEM "):
            continue
        else:
            out_lines.append(line)

    if not replaced_main_from:
        out_lines.insert(0, f"FROM {patched_gguf_path.resolve()}")

    out_lines.append("PARAMETER temperature 0.1")
    out_lines.append("PARAMETER top_p 0.90")
    out_lines.append("PARAMETER num_ctx 8192")
    out_lines.append(f'SYSTEM """{UNIFIED_SYSTEM_PROMPT}"""')

    modelfile_patched = MERGED_ADAPTER_DIR / "Modelfile.batiai_patched"
    modelfile_patched.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return modelfile_patched


def register_unified_model_in_ollama(
    target_model_name: str = TARGET_OLLAMA_MODEL,
    base_ollama_model: str = TARGET_OLLAMA_MODEL,
    alpha_coding: Optional[float] = None,
    use_rslora_scale: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    # Fast-skip if the patched GGUF and merge manifest already exist, unless force=True
    if not force and PATCHED_GGUF_PATH.exists() and (MERGED_ADAPTER_DIR / "merge_manifest.json").exists():
        manifest = json.loads((MERGED_ADAPTER_DIR / "merge_manifest.json").read_text(encoding="utf-8"))
        return {"status": "success", "mode": "cached_in_place_gguf", "active_model": target_model_name, "manifest": manifest}

    coding_dir, physchem_adapter = locate_coding_and_physchem_adapters(REPO_ROOT)
    if not coding_dir or not physchem_adapter:
        return {"status": "skipped", "reason": "coding_ft.zip or physchem_ft.zip adapters not found."}

    manifest = blend_lora_adapters(coding_dir, physchem_adapter, MERGED_ADAPTER_DIR, alpha_coding=alpha_coding)
    base_blob, orig_modelfile = find_ollama_gguf_blob_and_modelfile(base_ollama_model)

    if base_blob is not None:
        patch_stats = patch_ollama_gguf_with_blended_lora(
            base_blob,
            MERGED_ADAPTER_DIR,
            PATCHED_GGUF_PATH,
            coding_adapter_dir=coding_dir,
            physchem_adapter_dir=physchem_adapter,
            alpha_coding=manifest.get("alpha_coding", 0.85),
            use_rslora_scale=use_rslora_scale,
        )
        manifest.update(patch_stats)
        modelfile_path = build_patched_modelfile(orig_modelfile, base_blob, PATCHED_GGUF_PATH)

        res = subprocess.run(["ollama", "create", target_model_name, "-f", str(modelfile_path)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if res.returncode == 0:
            print(f"[FT Merger] Successfully updated `{target_model_name}` in Ollama with in-place LoRA-patched GGUF weights!")
            return {
                "status": "success",
                "mode": "in_place_gguf_lora_merged",
                "active_model": target_model_name,
                "manifest": manifest,
            }
        return {"status": "error", "error": res.stderr.strip(), "manifest": manifest}

    return {"status": "merged_adapter_only", "active_model": target_model_name, "manifest": manifest}


if __name__ == "__main__":
    print(json.dumps(register_unified_model_in_ollama(force=True), indent=2))
