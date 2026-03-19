import importlib.machinery
import platform
import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _ensure_vllm_stubs_for_macos() -> None:
    # Linux environments should use the real vLLM dependency.
    if platform.system() != "Darwin":
        return

    try:
        __import__("vllm")
        return
    except ModuleNotFoundError:
        pass

    for name in (
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

    class _LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    sys.modules["vllm.lora.request"].LoRARequest = _LoRARequest
    sys.modules["vllm.lora.utils"].get_adapter_absolute_path = lambda path: path
    sys.modules["vllm.multimodal"].MultiModalDataDict = dict

    tokenizer_mod = sys.modules["vllm.transformers_utils.tokenizer"]
    tokenizer_mod.AnyTokenizer = object
    tokenizer_mod.get_lora_tokenizer = lambda *args, **kwargs: None
    tokenizer_mod.get_tokenizer = lambda *args, **kwargs: None

    sys.modules["vllm.utils"].FlexibleArgumentParser = type(
        "FlexibleArgumentParser",
        (),
        {},
    )


_ensure_vllm_stubs_for_macos()
