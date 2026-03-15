import importlib.machinery
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _stub_optional_dependencies() -> None:
    for name in (
        "datasets",
        "pandas",
        "PIL",
        "vllm",
        "vllm.lora",
        "vllm.lora.request",
        "vllm.lora.utils",
        "vllm.multimodal",
        "vllm.transformers_utils",
        "vllm.transformers_utils.tokenizer",
        "vllm.utils",
    ):
        if name not in sys.modules:
            _stub_module(name)

    sys.modules["datasets"].load_dataset = lambda *args, **kwargs: None
    sys.modules["PIL"].Image = type("Image", (), {})
    sys.modules["vllm.lora.request"].LoRARequest = type("LoRARequest", (), {})
    sys.modules["vllm.lora.utils"].get_adapter_absolute_path = (
        lambda *args, **kwargs: ""
    )
    sys.modules["vllm.multimodal"].MultiModalDataDict = dict
    sys.modules["vllm.transformers_utils.tokenizer"].AnyTokenizer = object
    sys.modules["vllm.transformers_utils.tokenizer"].get_lora_tokenizer = (
        lambda *args, **kwargs: None
    )
    sys.modules["vllm.transformers_utils.tokenizer"].get_tokenizer = (
        lambda *args, **kwargs: None
    )
    sys.modules["vllm.utils"].FlexibleArgumentParser = type(
        "FlexibleArgumentParser",
        (),
        {},
    )


class _DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        _ = (text, add_special_tokens)
        return SimpleNamespace(input_ids=[1, 2, 3])


class TestBenchmarkServingRequestTimestamps(unittest.TestCase):
    def test_request_timestamps_aligned_with_request_arrays_for_mixed_outcomes(self):
        _stub_optional_dependencies()

        from backend_request_func import RequestFuncOutput
        from benchmark_serving import calculate_metrics

        input_requests = [
            SimpleNamespace(prompt_len=16),
            SimpleNamespace(prompt_len=24),
            SimpleNamespace(prompt_len=32),
        ]
        outputs = [
            RequestFuncOutput(
                success=True,
                output_tokens=8,
                ttft=0.10,
                itl=[0.02] * 7,
                latency=0.24,
                prompt_len=16,
                request_timestamp=1000.0,
            ),
            RequestFuncOutput(
                success=False,
                output_tokens=0,
                ttft=0.0,
                itl=[],
                latency=0.0,
                prompt_len=24,
                request_timestamp=1000.5,
                error="boom",
            ),
            RequestFuncOutput(
                success=True,
                output_tokens=4,
                ttft=0.08,
                itl=[0.03] * 3,
                latency=0.17,
                prompt_len=32,
                request_timestamp=1001.0,
            ),
        ]

        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=1.0,
            tokenizer=_DummyTokenizer(),
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            selected_percentiles=[50, 90],
            goodput_config_dict={},
        )

        self.assertEqual(len(metrics.request_timestamps), len(outputs))
        self.assertEqual(metrics.request_timestamps, [1000.0, 1000.5, 1001.0])
        self.assertEqual(actual_output_lens, [8, 0, 4])


if __name__ == "__main__":
    unittest.main()
