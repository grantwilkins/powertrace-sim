"""Architecture registry: one home for model architecture descriptors (D3).

The ARCH dict is moved VERBATIM from ``feature-test/build_ledger_cache.py``.
Its insertion order is load-bearing: the ledger cache encodes
``model_idx = list(ARCH).index(model)``, so entries must only ever be
appended, never reordered.

Bundles record their arch in-band (``manifest.json`` ``arch`` block, produced
by ``profiling/client/arch_extract.py``); this registry serves the legacy
layouts whose identity lives only in file names.
"""

from __future__ import annotations

import math
from typing import Dict

GIB = 1024.0**3


def _dense_components(
    *,
    n_active: float,
    w_bytes: float,
    vocab_size: int,
    d_model: int,
    embedding_bytes_per_param: float = 2.0,
) -> Dict[str, object]:
    embedding = float(vocab_size * d_model)
    embedding_bytes = embedding_bytes_per_param * embedding
    return {
        "transformer_active_params": n_active - 2.0 * embedding,
        "input_embedding_params": embedding,
        "output_head_params": embedding,
        "transformer_weight_bytes": w_bytes - 2.0 * embedding_bytes,
        "input_embedding_weight_bytes": embedding_bytes,
        "output_head_weight_bytes": embedding_bytes,
        "tied_embeddings": 0,
        "vocab_size": vocab_size,
    }


# Architecture descriptors (as served; see profiling/server/*.sh for dtypes).
ARCH = {
    "llama-3-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
        **_dense_components(
            n_active=8.03e9, w_bytes=2 * 8.03e9,
            vocab_size=128256, d_model=4096,
        ),
    ),
    "deepseek-r1-distill-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
        **_dense_components(
            n_active=8.03e9, w_bytes=2 * 8.03e9,
            vocab_size=128256, d_model=4096,
        ),
    ),
    "llama-3-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
        **_dense_components(
            n_active=70.55e9, w_bytes=2 * 70.55e9,
            vocab_size=128256, d_model=8192,
        ),
    ),
    "deepseek-r1-distill-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
        **_dense_components(
            n_active=70.55e9, w_bytes=2 * 70.55e9,
            vocab_size=128256, d_model=8192,
        ),
    ),
    # Served checkpoint is meta-llama/Llama-3.1-405B-Instruct-FP8
    # (profiling/server/serve-llama-3-405b.sh). Its FP8 recipe quantizes FFN
    # matmuls only — not attention, not the first/last layers, not
    # embeddings/lm_head (arXiv:2407.21783 section 6.2) — so the resident
    # weight bytes are the checkpoint total 487,229,436,720 B (sum of the
    # 109 safetensors shards from the HF tree API; architecture-derived
    # estimate reconciles to 177 KB). fp8_flop_frac is the FP8 parameter
    # share 324.538e9 / 405.853e9: FLOPs per token are proportional to
    # params touched, so this is also the FP8 FLOP fraction.
    "llama-3-405b": dict(
        family="dense-405b", n_active=405.85e9, w_bytes=487.23e9, d_model=16384,
        n_layers=126, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=1, fp8_flop_frac=0.7996,
        **_dense_components(
            n_active=405.85e9, w_bytes=487.23e9,
            vocab_size=128256, d_model=16384,
        ),
    ),
    "gpt-oss-120b": dict(
        family="moe-120b", n_active=5.1e9, w_bytes=60.8 * GIB, d_model=2880,
        n_layers=36, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=128, top_k=4,
        swa_window=128, fp8=0,
    ),
    "gpt-oss-20b": dict(
        family="moe-20b", n_active=3.6e9, w_bytes=12.8 * GIB, d_model=2880,
        n_layers=24, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=32, top_k=4,
        swa_window=128, fp8=0,
    ),
}

MANIFEST_COMPONENT_ARCH = {
    "Qwen/Qwen3-8B": {
        "n_active": 8190427136.0,
        **_dense_components(
            n_active=8190427136.0, w_bytes=16380854272.0,
            vocab_size=151936, d_model=4096,
        ),
    },
}


def get_arch(model: str) -> Dict[str, object]:
    """Registry lookup for legacy (name-identified) runs."""
    try:
        return ARCH[model]
    except KeyError:
        raise KeyError(
            f"Unknown model '{model}' in architecture registry; "
            f"known: {sorted(ARCH)}"
        ) from None


def arch_from_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    """Arch block from a bundle manifest (recorded in-band by arch_extract)."""
    arch = manifest.get("arch")
    if not isinstance(arch, dict) or not arch:
        raise ValueError("Bundle manifest has no 'arch' block")
    reference = MANIFEST_COMPONENT_ARCH.get(str(manifest.get("model", "")))
    if reference and "transformer_active_params" not in arch:
        out = dict(arch)
        if not math.isclose(
            float(out["n_active"]), float(reference["n_active"]), rel_tol=1e-3
        ):
            raise ValueError("Manifest architecture does not match its component descriptor")
        for key in (
            "transformer_active_params",
            "input_embedding_params",
            "output_head_params",
            "transformer_weight_bytes",
            "input_embedding_weight_bytes",
            "output_head_weight_bytes",
            "tied_embeddings",
            "vocab_size",
        ):
            out[key] = reference[key]
        return out
    return arch

