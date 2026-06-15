"""How much data does the model need? Learning curve + coverage test.

Hold out dense-70B (H100) entirely as the test set. Fit hardware constants on
shrinking subsets of the OTHER H100 models (8B, 405B), then predict 70B
zero-shot. Tests whether what matters is sample COUNT or load-range COVERAGE.

Run: uv run python feature-test/data_efficiency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

import fit_map_priors as fmp  # noqa: E402
from peak_and_holdout import FEATS2, IDLE2, design2, setup_features  # noqa: E402
from fit_models import load_cache, metrics  # noqa: E402


def main():
    d = load_cache()
    setup_features(d)
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    rates = d["rate"]
    tps = d["tp"]
    mn = [str(x) for x in d["model_names"]]
    fam_names = [str(x) for x in d["family_names"]]
    idle_cols = np.array([f in IDLE2 for f in FEATS2])

    H = d["hw_idx"] == 1
    X = design2(d, "H100", run_ids)
    fi70 = fam_names.index("dense-70b")
    test = H & (fams == fi70)
    is8 = H & np.array([mn[i] == "llama-3-8b" for i in d["model_idx"]])

    # bin index within each run (for time-budget slicing)
    pos = np.zeros(y.size, dtype=int)
    for r in np.unique(run_ids):
        idx = np.flatnonzero(run_ids == r)
        pos[idx] = np.arange(idx.size)

    def first_run_per_cell(mask):
        keep = np.zeros(y.size, bool)
        seen = set()
        for r in np.unique(run_ids[mask]):
            i0 = np.flatnonzero(run_ids == r)[0]
            cell = (int(d["model_idx"][i0]), int(tps[i0]), float(rates[i0]))
            if cell in seen:
                continue
            seen.add(cell)
            keep |= (run_ids == r)
        return keep & mask

    def evaluate(train_mask, label):
        th, mult, _ = fmp.fit_two_stage(X[train_mask], y[train_mask], fams[train_mask], idle_cols)
        busy = train_mask & (d["batch"] > 0)
        cap = float(np.quantile(y[busy] / tps[busy], 0.995)) if busy.sum() else 600.0
        pred = np.minimum(fmp.predict_map(X[test], th, {}, fams[test], idle_cols), cap * tps[test])
        errs = [abs(pred[run_ids[test] == r].mean() / y[test][run_ids[test] == r].mean() - 1) * 100
                for r in np.unique(run_ids[test])]
        secs = sum(int((run_ids == r).sum()) for r in np.unique(run_ids[train_mask]))  # 1s bins
        nruns = len(np.unique(run_ids[train_mask]))
        print(f"  {label:48s} | train: {nruns:3d} runs, {secs:6d} GPU-bin-s | "
              f"held-out 70B med {np.median(errs):5.1f}%  worst {max(errs):5.1f}%  cap {cap:.0f}W")
        return np.median(errs)

    print("=== LEARNING CURVE: held-out dense-70B (H100), trained on 8B+405B ===")
    pool = H & (fams != fi70)
    evaluate(pool, "full pool (8B+405B, all rates, 3 runs/cell)")
    evaluate(first_run_per_cell(pool), "1 run per cell")

    print("\n=== SINGLE MODEL (llama-3-8B only), TP4+TP8 (the TPs 70B uses) ===")
    base = is8 & ((tps == 4) | (tps == 8))
    evaluate(first_run_per_cell(base), "8B, all 6 rates, 1 run each")

    print("\n=== COVERAGE vs COUNT (8B, TP4+8, 1 run each) ===")
    def rate_subset(rset):
        m = np.zeros(y.size, bool)
        for rr in rset:
            m |= (base & (rates == rr))
        return first_run_per_cell(m)
    evaluate(rate_subset([0.125, 4.0]), "WIDE: 2 rates {min 0.125, max 4}")
    evaluate(rate_subset([0.125, 0.25]), "NARROW: 2 low rates {0.125, 0.25}")
    evaluate(rate_subset([1.0]), "SINGLE rate {1.0}")
    evaluate(rate_subset([0.125, 0.5, 4.0]), "WIDE+: 3 rates {0.125, 0.5, 4}")

    print("\n=== TIME BUDGET (8B, TP4+8, wide 2 rates, first N seconds of each run) ===")
    wide = rate_subset([0.125, 4.0])
    for nsec in (30, 60, 120, 300):
        evaluate(wide & (pos < nsec), f"first {nsec}s of each run")


if __name__ == "__main__":
    main()
