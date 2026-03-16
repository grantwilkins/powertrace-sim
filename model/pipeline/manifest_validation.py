from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


_SCHEMAS: Dict[str, Dict[str, Tuple[type, ...]]] = {
    "run_manifest": {
        "configs": (dict,),
    },
    "experimental_manifest": {
        "configs": (dict,),
    },
    "throughput_db": {
        "configs": (dict,),
    },
    "split_manifest": {
        "train_indices": (list,),
        "val_indices": (list,),
        "test_indices": (list,),
    },
}


def validate_manifest(payload: Mapping[str, Any], schema_name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{schema_name} must be a JSON object")
    schema = _SCHEMAS.get(str(schema_name))
    if schema is None:
        raise ValueError(f"unknown manifest schema: {schema_name}")
    missing = [key for key in schema if key not in payload]
    if missing:
        raise ValueError(f"{schema_name} missing required keys: {missing}")
    for key, expected_types in schema.items():
        value = payload.get(key)
        if not isinstance(value, expected_types):
            names = ", ".join(t.__name__ for t in expected_types)
            raise ValueError(
                f"{schema_name}.{key} must be {names}; got {type(value).__name__}"
            )


__all__ = ["validate_manifest"]
