"""Frozen primary gates and source-only candidate selection."""

from statistics import median

PREFERENCE = ("M0", "M0b", "M0bR", "M0dR", "M0c", "M4A", "M1", "M2", "M3")


def energy_limit(split_name: str) -> float:
    return 5.0 if split_name.startswith(("S0", "S1", "S2a")) else 7.5


def passes_primary(score: dict, split_name: str) -> bool:
    acf, mae, nrmse, e_p90, e_worst = 0.75, 0.12, 0.17, 15.0, 20.0
    if split_name.startswith("S0"):
        acf, mae, nrmse, e_p90, e_worst = 0.85, 0.10, 0.15, 10.0, 15.0
    elif split_name.startswith("S1"):
        acf, nrmse = 0.80, 0.13
    return bool(
        score["energy_error_pct_median"] <= energy_limit(split_name)
        and score["energy_error_pct_p90"] <= e_p90
        and (split_name.startswith("S3") or score["energy_error_pct_worst"] <= e_worst)
        and score["acf_r2_median"] >= acf
        and score["acf_mae_p90"] <= mae
        and score["nrmse_range_median"] <= nrmse
    )


def passes_correction_safety(score: dict, physics: dict) -> bool:
    """Apply G5 to one correction and its physics-only reference."""
    dynamics = (
        score["acf_r2_median"] >= physics["acf_r2_median"] + 0.05
        or score["acf_mae_median"] <= 0.8 * physics["acf_mae_median"]
    )
    cap_safe = (
        score["cap_hit_fraction_median"] <= physics["cap_hit_fraction_median"] + 0.01
        or score["energy_error_pct_median"] <= physics["energy_error_pct_median"]
    )
    return bool(
        score["energy_error_pct_median"] <= physics["energy_error_pct_median"] + 0.5
        and score["energy_error_pct_p90"] <= physics["energy_error_pct_p90"] + 1.0
        and dynamics
        and score["nrmse_range_median"] <= physics["nrmse_range_median"]
        and cap_safe
    )


def passes_b2_comparison(score: dict, baseline: dict) -> bool:
    """Apply the frozen G1 tolerance relative to same-configuration B2."""
    return bool(
        score["energy_error_pct_median"] <= baseline["energy_error_pct_median"] + 1.0
        and score["acf_r2_median"] >= baseline["acf_r2_median"] - 0.05
        and score["nrmse_range_median"] <= baseline["nrmse_range_median"] + 0.02
    )


def choose_hardware_candidate(split_names, source_scores, physics_bases) -> str:
    """Choose one hardware mode from all of its source-development cells."""
    choices = []
    for candidate in PREFERENCE:
        scores = [(name, source_scores[name][candidate]) for name in split_names]
        unsafe = sum(
            candidate in ("M1", "M2", "M3") and not passes_correction_safety(
                score, source_scores[name][physics_bases[name]])
            for name, score in scores
        )
        choices.append((sum(not passes_primary(score, name) for name, score in scores) + unsafe,
                        median(score["energy_error_pct_median"] for _, score in scores),
                        -median(score["acf_r2_median"] for _, score in scores),
                        median(score["nrmse_range_median"] for _, score in scores),
                        PREFERENCE.index(candidate), candidate))
    return min(choices)[-1]


def choose_source_candidate(dev_scores: dict[str, dict], split_name: str) -> str:
    """Freeze the smallest passing candidate using source development only."""
    passing = [name for name in PREFERENCE if passes_primary(dev_scores[name], split_name)]
    if passing:
        return passing[0]
    return min(PREFERENCE, key=lambda name: (
        dev_scores[name]["energy_error_pct_median"] > energy_limit(split_name),
        dev_scores[name]["energy_error_pct_median"] if
        dev_scores[name]["energy_error_pct_median"] > energy_limit(split_name) else 0.0,
        -dev_scores[name]["acf_r2_median"], dev_scores[name]["nrmse_range_median"],
        PREFERENCE.index(name),
    ))
