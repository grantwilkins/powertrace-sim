import importlib.machinery
import platform
import sys
import types
from pathlib import Path

# profiling/probes (schedule, runner) and profiling/client (loggers, manifest).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "client"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _ensure_vllm_stubs_for_macos() -> None:
    if platform.system() != "Darwin":
        return
    try:
        __import__("vllm")
        return
    except ModuleNotFoundError:
        pass
    for name in (
        "vllm", "vllm.lora", "vllm.lora.request", "vllm.lora.utils",
        "vllm.multimodal", "vllm.transformers_utils",
        "vllm.transformers_utils.tokenizer", "vllm.utils",
    ):
        if name not in sys.modules:
            _stub_module(name)

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    sys.modules["vllm.lora.request"].LoRARequest = _LoRARequest
    sys.modules["vllm.lora.utils"].get_adapter_absolute_path = lambda path: path
    sys.modules["vllm.multimodal"].MultiModalDataDict = dict
    tok = sys.modules["vllm.transformers_utils.tokenizer"]
    tok.AnyTokenizer = object
    tok.get_lora_tokenizer = lambda *a, **k: None
    tok.get_tokenizer = lambda *a, **k: None
    sys.modules["vllm.utils"].FlexibleArgumentParser = type(
        "FlexibleArgumentParser", (), {})


_ensure_vllm_stubs_for_macos()
