"""
Claim:
Iteration work is hand-computable from architecture descriptors alone
(dense and mixture-of-experts, decode and causal prefill chunks, sliding
window), and iteration time follows the roofline rule with work divided
across GPUs plus a launch overhead.

Plausible wrong implementations:
- Charge a full weight sweep for decode and prefill separately in a mixed
  iteration (double sweep).
- Use raw context where sliding-window layers cap it.
- Forget the FP8 FLOP-demand scale or apply it to bytes.
- Divide the launch overhead by GPU count.
- Use chunk-end context instead of the causal mean for prefill attention.
- Charge the input embedding matrix as a dense per-token operator.
- Collapse independent prefill chunks into one fictitious attention context.
- Collapse sequential transformer and output-head operators under one roofline.
- Apply an FP8 transformer calibration to BF16 head, attention, or KV traffic.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from iteration_time import (  # noqa: E402
    expected_weight_bytes_per_sweep,
    iteration_time_s,
    iteration_work,
    kv_bytes_per_token,
    transformer_bw_scale,
)
from model.training_data.moe_routing import RoutingLaw  # noqa: E402

DENSE = dict(n_active=10.0, w_bytes=100.0, d_model=8.0, n_layers=2, n_kv=1,
             head_dim=4, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0, fp8=0)
MOE = dict(DENSE, moe_frac=0.5, n_experts=4, top_k=1)
COMPONENT_DENSE = dict(
    DENSE,
    transformer_active_params=6.0,
    input_embedding_params=20.0,
    output_head_params=2.0,
    transformer_weight_bytes=60.0,
    input_embedding_weight_bytes=20.0,
    output_head_weight_bytes=20.0,
)


def test_dense_decode_work_is_hand_computed():
    # kv_tok = 2*2*1*4*2 = 32 bytes/token; n_q = 8/4 = 2.
    assert kv_bytes_per_token(DENSE) == 32.0
    work = iteration_work(DENSE, decode_batch=3, context_mean=5)
    assert work["gemm_flops"] == 60.0            # 2*10*3
    assert work["attn_flops"] == 960.0           # 4*2*4*2*3*5
    assert work["gemm_bytes"] == 100.0           # one weight sweep
    assert work["attn_bytes"] == 3 * (5 + 1) * 32.0  # KV read + write


def test_mixed_iteration_charges_one_weight_sweep():
    decode_only = iteration_work(DENSE, decode_batch=2, context_mean=0)
    mixed = iteration_work(DENSE, decode_batch=2, context_mean=0,
                           prefill_chunk=4, prefill_context=0)
    # Chunk adds: GEMM FLOPs 2*10*4; attention FLOPs over causal mean
    # context 2: 4*2*4*2*4*2 = 512; attention bytes: ONE streaming pass of
    # the visible context (2*32, kernels tile KV across the chunk's
    # queries) plus KV write 4*32. The weight sweep is charged once.
    assert mixed["gemm_flops"] == decode_only["gemm_flops"] + 80.0
    assert mixed["attn_flops"] == decode_only["attn_flops"] + 512.0
    assert mixed["attn_bytes"] == decode_only["attn_bytes"] + 64.0 + 128.0
    assert mixed["gemm_bytes"] == 100.0


def test_component_work_treats_embedding_as_rows_and_output_as_projection():
    prefill = iteration_work(
        COMPONENT_DENSE, prefill_chunk=4, prefill_context=0,
        prefill_logits=1,
    )
    # Transformer: 2*6*4. One final-token output projection: 2*2.
    assert prefill["gemm_flops"] == 52.0
    # Transformer sweep 60 + output-head sweep 20 + four embedding rows
    # of d_model=8 at one byte/parameter.
    assert prefill["gemm_bytes"] == 112.0
    assert prefill["transformer_flops"] == 48.0
    assert prefill["output_head_flops"] == 4.0
    assert prefill["transformer_bytes"] == 60.0
    assert prefill["output_head_bytes"] == 20.0
    assert prefill["embedding_bytes"] == 32.0


def test_nonfinal_prefill_is_vocab_invariant_but_final_head_is_not():
    small = dict(COMPONENT_DENSE)
    large = dict(
        COMPONENT_DENSE,
        output_head_params=20.0,
        output_head_weight_bytes=200.0,
    )
    nonfinal_small = iteration_work(small, prefill_chunk=4)
    nonfinal_large = iteration_work(large, prefill_chunk=4)
    assert nonfinal_small["gemm_flops"] == nonfinal_large["gemm_flops"]
    assert nonfinal_small["gemm_bytes"] == nonfinal_large["gemm_bytes"]
    final_small = iteration_work(small, prefill_chunk=4, prefill_logits=1)
    final_large = iteration_work(large, prefill_chunk=4, prefill_logits=1)
    assert final_large["output_head_flops"] > final_small["output_head_flops"]
    assert final_large["output_head_bytes"] > final_small["output_head_bytes"]


def test_independent_prefill_chunks_recompose_attention_work():
    combined = iteration_work(
        DENSE, prefill_chunks=[(1, 0), (3, 100)],
    )
    separate = [
        iteration_work(DENSE, prefill_chunk=tokens, prefill_context=context)
        for tokens, context in ((1, 0), (3, 100))
    ]
    assert combined["attn_flops"] == pytest.approx(
        sum(work["attn_flops"] for work in separate)
    )
    assert combined["attn_bytes"] == pytest.approx(
        sum(work["attn_bytes"] for work in separate)
    )


def test_layer_scaled_launch_overhead():
    from iteration_time import launch_overhead_s
    assert launch_overhead_s(DENSE, base_s=0.001, per_message_s=0.0001) == \
        pytest.approx(0.001 + 2 * 2 * 0.0001)


def test_moe_expected_sweep_lands_in_gemm_bytes():
    assert iteration_work(MOE, decode_batch=1, context_mean=0)["gemm_bytes"] \
        == expected_weight_bytes_per_sweep(MOE, 1)


def test_moe_mixed_work_uses_phase_aware_routing_union():
    law = RoutingLaw(
        model="toy", source="toy", top_k=1, n_experts=4,
        prefill_alpha=1.0, decode_alpha=1.0,
        prefill_touch_probability=np.full((1, 4), 0.25),
        decode_touch_probability=np.full((1, 4), 0.25),
    )
    work = iteration_work(
        MOE, decode_batch=1, prefill_chunk=1, routing_law=law)
    # Each expert is touched with probability 1 - .75*.75 = 7/16.
    assert work["gemm_bytes"] == pytest.approx(
        100.0 * (0.5 + 0.5 * 7.0 / 16.0))


def test_sampling_term_is_batch_linear_and_not_divided_by_gpus():
    work = iteration_work(DENSE, decode_batch=256, context_mean=1)
    assert work["sampled_tokens"] == 256
    base = iteration_time_s(work, hardware="A100", tp=8, eff_flops=1.0,
                            eff_bw=1.0, t_launch_s=0.0)
    with_sampling = iteration_time_s(work, hardware="A100", tp=8,
                                     eff_flops=1.0, eff_bw=1.0,
                                     t_launch_s=0.0, t_sample_s=4e-5)
    assert with_sampling - base == pytest.approx(256 * 4e-5)


def test_moe_expected_sweep_saturates_with_tokens():
    assert expected_weight_bytes_per_sweep(MOE, 1) == 100.0 * (0.5 + 0.5 * 0.25)
    nearly_all = expected_weight_bytes_per_sweep(MOE, 1000)
    assert 99.9 < nearly_all <= 100.0
    assert expected_weight_bytes_per_sweep(MOE, 0) == 0.0


def test_sliding_window_caps_effective_context():
    swa = dict(DENSE, swa_window=4)
    capped = iteration_work(swa, decode_batch=1, context_mean=100)
    uncapped = iteration_work(DENSE, decode_batch=1, context_mean=100)
    # effective context: 0.5*100 + 0.5*4 = 52 vs 100
    assert capped["attn_flops"] == pytest.approx(
        uncapped["attn_flops"] - 4 * 2 * 4 * 2 * 1 * (100 - 52))
    assert capped["attn_bytes"] == pytest.approx(
        uncapped["attn_bytes"] - (100 - 52) * 32)


def test_fp8_fraction_scales_flops_not_bytes():
    fp8 = dict(COMPONENT_DENSE, fp8=1, fp8_flop_frac=0.8)
    base = iteration_work(COMPONENT_DENSE, decode_batch=2, context_mean=5)
    scaled = iteration_work(fp8, decode_batch=2, context_mean=5)
    assert scaled["transformer_flops"] == pytest.approx(
        0.6 * base["transformer_flops"]
    )
    assert scaled["output_head_flops"] == base["output_head_flops"]
    assert scaled["attn_flops"] == base["attn_flops"]
    assert scaled["gemm_bytes"] == base["gemm_bytes"]
    assert scaled["attn_bytes"] == base["attn_bytes"]


def test_fp8_calibration_is_required_only_for_fp8():
    assert transformer_bw_scale(DENSE, {}, "A100") == 1.0
    with pytest.raises(ValueError, match="FP8 timing requires"):
        transformer_bw_scale(dict(DENSE, fp8=1), {}, "H100")
    assert transformer_bw_scale(
        dict(DENSE, fp8=1), {"fp8_stream_scale": 0.75}, "H100"
    ) == 0.75


def _work(gemm_flops=0.0, gemm_bytes=0.0, attn_flops=0.0, attn_bytes=0.0):
    return {"gemm_flops": gemm_flops, "gemm_bytes": gemm_bytes,
            "attn_flops": attn_flops, "attn_bytes": attn_bytes}


def test_roofline_time_sums_operator_classes():
    # GEMM compute-bound (1 s at tp1) + attention memory-bound (0.5 s).
    work = _work(gemm_flops=312e12, gemm_bytes=1e12, attn_bytes=1e12)
    t1 = iteration_time_s(work, hardware="A100", tp=1, eff_flops=1.0,
                          eff_bw=1.0, t_launch_s=0.002)
    t4 = iteration_time_s(work, hardware="A100", tp=4, eff_flops=1.0,
                          eff_bw=1.0, t_launch_s=0.002)
    assert t1 == pytest.approx(0.002 + 1.0 + 0.5)    # classes ADD
    assert t4 == pytest.approx(0.002 + 0.25 + 0.125)  # work splits, launch not
    memory_bound = iteration_time_s(_work(gemm_flops=1e12, gemm_bytes=2e12),
                                    hardware="A100", tp=1, eff_flops=1.0,
                                    eff_bw=0.5, t_launch_s=0.0)
    assert memory_bound == pytest.approx(2e12 / (0.5 * 2e12))


def test_roofline_sums_transformer_head_and_embedding_classes():
    work = {
        "gemm_flops": 312e12,
        "gemm_bytes": 3e12,
        "transformer_flops": 312e12,
        "transformer_bytes": 0.0,
        "output_head_flops": 0.0,
        "output_head_bytes": 2e12,
        "embedding_bytes": 1e12,
        "attn_flops": 0.0,
        "attn_bytes": 0.0,
    }
    seconds = iteration_time_s(
        work, hardware="A100", tp=1, eff_flops=1.0, eff_bw=1.0,
        t_launch_s=0.0, transformer_bw_scale=0.25,
    )
    assert seconds == pytest.approx(1.0 + 1.0 + 0.5)
