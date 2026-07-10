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

from typing import Dict

GIB = 1024.0**3

# Architecture descriptors (as served; see profiling/server/*.sh for dtypes).
ARCH = {
    "llama-3-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "deepseek-r1-distill-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "llama-3-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "deepseek-r1-distill-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "llama-3-405b": dict(
        family="dense-405b", n_active=405.85e9, w_bytes=1 * 405.85e9, d_model=16384,
        n_layers=126, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=1,
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
    return arch
