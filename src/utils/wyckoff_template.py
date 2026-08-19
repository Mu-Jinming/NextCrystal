# path: src/utils/wyckoff_template.py
from __future__ import annotations

import json
from pathlib import Path


NUM_SPACE_GROUPS = 230


def load_multiplicity_dict(template_path: str | Path) -> dict[int, list[int]]:
    template_path = Path(template_path)
    with template_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    multiplicity_dict: dict[int, list[int]] = {}
    for group in data["space_groups"]:
        number = int(group["number"])
        multiplicities = [int(position["multiplicity"]) for position in group["wyckoff_positions"]]
        multiplicity_dict.setdefault(number, []).extend(multiplicities)
    return multiplicity_dict


def load_space_group_symbols(template_path: str | Path) -> dict[int, str]:
    """Load and validate the canonical symbol for every space-group number."""
    template_path = Path(template_path)
    with template_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = data.get("space_groups")
    if not isinstance(groups, list):
        raise ValueError(f"Invalid Wyckoff template (missing space_groups list): {template_path}")

    number_to_symbol: dict[int, str] = {}
    seen_symbols: set[str] = set()
    for group in groups:
        number = int(group["number"])
        symbol = str(group["symbol"]).strip()
        if number in number_to_symbol:
            raise ValueError(f"Duplicate space-group number {number} in {template_path}")
        if symbol in seen_symbols:
            raise ValueError(f"Duplicate space-group symbol {symbol!r} in {template_path}")
        number_to_symbol[number] = symbol
        seen_symbols.add(symbol)

    expected = set(range(1, NUM_SPACE_GROUPS + 1))
    actual = set(number_to_symbol)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Wyckoff template must contain space groups 1-{NUM_SPACE_GROUPS}; "
            f"missing={missing}, extra={extra}"
        )

    return dict(sorted(number_to_symbol.items()))


def build_space_group_vocab(
    template_path: str | Path,
    *,
    include_unknown: bool = False,
) -> dict[str, int]:
    """Build a deterministic vocabulary covering all 230 space groups.

    Class index ``i`` always represents international space-group number
    ``i + 1``.  This keeps the classifier label space independent of which
    groups happen to occur in a particular training split.
    """
    number_to_symbol = load_space_group_symbols(template_path)
    vocab = {symbol: number - 1 for number, symbol in number_to_symbol.items()}
    if include_unknown:
        vocab["<unk>"] = NUM_SPACE_GROUPS
    return vocab


def space_group_number_to_index(number: int) -> int:
    number = int(number)
    if not 1 <= number <= NUM_SPACE_GROUPS:
        raise ValueError(
            f"Space-group number must be in [1, {NUM_SPACE_GROUPS}], got {number}"
        )
    return number - 1


def get_multiplicity_vocab_size(multiplicity_dict: dict[int, list[int]]) -> int:
    if not multiplicity_dict:
        return 1
    return max(max(values, default=0) for values in multiplicity_dict.values())
