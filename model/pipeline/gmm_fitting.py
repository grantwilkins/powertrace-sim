from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from model.classifiers.gmm_bigru import fit_power_gmm

BRACKET_CONFIG_SUBSET = [
    "deepseek-r1-distill-8b_A100_tp1",
    "deepseek-r1-distill-70b_H100_tp8",
    "gpt-oss-20b_A100_tp2",
    "gpt-oss-120b_H100_tp8",
    "llama-3-8b_A100_tp1",
    "llama-3-8b_H100_tp8",
    "llama-3-70b_A100_tp4",
    "llama-3-405b_H100_tp8",
]


def _parse_k_candidates(csv_text: str) -> List[int]:
    out: List[int] = []
    for tok in str(csv_text).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        val = int(tok)
        if val < 1:
            continue
        out.append(val)
    deduped: List[int] = []
    seen = set()
    for val in out:
        if val in seen:
            continue
        deduped.append(val)
        seen.add(val)
    return deduped


def _fit_candidate_scores(
    power_values: np.ndarray,
    *,
    k_candidates: Sequence[int],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in k_candidates:
        kc = int(k)
        if kc < 1:
            continue
        if power_values.size < kc:
            out[str(kc)] = {"aic": float("nan"), "bic": float("nan")}
            continue
        try:
            fitted = fit_power_gmm(
                power_values=power_values,
                k=kc,
                random_state=int(seed),
                n_init=10,
                max_iter=300,
                reg_covar=1e-6,
            )
            out[str(kc)] = {
                "aic": float(fitted["aic"]),
                "bic": float(fitted["bic"]),
            }
        except Exception:
            out[str(kc)] = {"aic": float("nan"), "bic": float("nan")}
    return out


def _select_optimal_k(
    bic_scores: Dict[str, Dict[str, float]],
    *,
    max_k: int = 20,
    min_k: int = 4,
) -> Tuple[int, str]:
    """
    Select optimal K based on BIC scores.

    Returns:
        Tuple of (selected_k, selection_reason)
    """
    valid_scores: Dict[int, float] = {}
    for k_str, scores in bic_scores.items():
        k_val = int(k_str)
        bic_val = scores.get("bic", float("nan"))
        if np.isfinite(bic_val) and min_k <= k_val <= max_k:
            valid_scores[k_val] = bic_val

    if not valid_scores:
        return max(min_k, 10), "no_valid_bic_scores_fallback"

    # Find K with minimum BIC
    best_k = min(valid_scores, key=lambda k: valid_scores[k])

    # Check if BIC is still decreasing at max tested K (suggests higher K might be better)
    sorted_ks = sorted(valid_scores.keys())
    if len(sorted_ks) >= 2:
        max_tested_k = sorted_ks[-1]
        second_max_k = sorted_ks[-2]
        if (
            best_k == max_tested_k
            and valid_scores[max_tested_k] < valid_scores[second_max_k]
        ):
            reason = f"bic_minimum_at_max_k={best_k}_may_need_higher"
        else:
            reason = f"bic_minimum_at_k={best_k}"
    else:
        reason = f"bic_minimum_at_k={best_k}"

    return best_k, reason
