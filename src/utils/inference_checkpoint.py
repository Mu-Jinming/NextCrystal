from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT = "nextcrystal-inference"
CHECKPOINT_VERSION = 1
SUPPORTED_TASKS = {"spacegroup", "wyckoff"}


def _as_ordered_state_dict(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict must be a non-empty mapping")

    normalized = OrderedDict((str(key), value) for key, value in state.items())
    for prefix in ("model.", "module."):
        if all(key.startswith(prefix) for key in normalized):
            normalized = OrderedDict(
                (key[len(prefix) :], value) for key, value in normalized.items()
            )
    return normalized


def extract_model_state_dict(checkpoint: Any) -> OrderedDict[str, Any]:
    """Return a model-only state dict from a release, Lightning, or legacy file."""
    state = checkpoint
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    if not isinstance(state, Mapping):
        raise ValueError("Unsupported checkpoint: no state_dict mapping found")

    # Lightning task checkpoints contain model.*, metric.*, and criterion.*
    # entries.  Only the wrapped neural network belongs in an inference file.
    model_entries = OrderedDict(
        (str(key)[len("model.") :], value)
        for key, value in state.items()
        if str(key).startswith("model.")
    )
    if model_entries:
        return _as_ordered_state_dict(model_entries)

    return _as_ordered_state_dict(state)


def _validate_vocab(name: str, vocab: Any) -> dict[str, int]:
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"{name} must be a non-empty dict")
    if not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in vocab.items()
    ):
        raise ValueError(f"{name} must map string tokens to integer indices")
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError(f"{name} indices must be dense in [0, {len(vocab) - 1}]")
    return dict(vocab)


def is_release_checkpoint(checkpoint: Any) -> bool:
    return (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("format") == CHECKPOINT_FORMAT
    )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_task: str | None = None,
) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not is_release_checkpoint(checkpoint):
        return checkpoint

    version = checkpoint.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version {version!r} in {path}; "
            f"expected {CHECKPOINT_VERSION}"
        )

    task = checkpoint.get("task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported checkpoint task {task!r} in {path}")
    if expected_task is not None and task != expected_task:
        raise ValueError(
            f"Checkpoint task mismatch for {path}: expected {expected_task!r}, "
            f"found {task!r}"
        )

    extract_model_state_dict(checkpoint)
    _validate_vocab("formula_vocab", checkpoint.get("formula_vocab"))
    _validate_vocab("space_group_vocab", checkpoint.get("space_group_vocab"))
    if task == "wyckoff":
        _validate_vocab("wyckoff_vocab", checkpoint.get("wyckoff_vocab"))
    return checkpoint


def get_checkpoint_vocab(checkpoint: Any, name: str) -> dict[str, int] | None:
    if not is_release_checkpoint(checkpoint):
        return None
    value = checkpoint.get(name)
    if value is None:
        return None
    return _validate_vocab(name, value)


def get_checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not is_release_checkpoint(checkpoint):
        return {}
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")
    return dict(metadata)


def save_inference_checkpoint(
    path: str | Path,
    *,
    task: str,
    state_dict: Mapping[str, Any],
    formula_vocab: dict[str, int],
    space_group_vocab: dict[str, int],
    wyckoff_vocab: dict[str, int] | None = None,
    model_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one self-contained checkpoint for an inference stage."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported checkpoint task: {task!r}")
    if task == "wyckoff" and wyckoff_vocab is None:
        raise ValueError("wyckoff_vocab is required for a Wyckoff checkpoint")

    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "task": task,
        "state_dict": _as_ordered_state_dict(state_dict),
        "formula_vocab": _validate_vocab("formula_vocab", formula_vocab),
        "space_group_vocab": _validate_vocab(
            "space_group_vocab", space_group_vocab
        ),
        "model_config": dict(model_config or {}),
        "metadata": dict(metadata or {}),
    }
    if wyckoff_vocab is not None:
        payload["wyckoff_vocab"] = _validate_vocab(
            "wyckoff_vocab", wyckoff_vocab
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_model_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
) -> None:
    """Load release and legacy weights, including DataParallel prefixes."""
    state = extract_model_state_dict(checkpoint)
    candidates = [state]
    if any(key.startswith("module.") for key in state):
        candidates.append(
            OrderedDict(
                (
                    key[len("module.") :] if key.startswith("module.") else key,
                    value,
                )
                for key, value in state.items()
            )
        )
    else:
        candidates.append(
            OrderedDict((f"module.{key}", value) for key, value in state.items())
        )

    errors: list[str] = []
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as error:
            errors.append(str(error))

    raise RuntimeError(
        "Checkpoint weights do not match the configured model. "
        + " | ".join(errors)
    )
