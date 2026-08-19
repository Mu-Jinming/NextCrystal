# path: src/utils/chemistry.py
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable


ELEMENT_ORDER = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_ORDER_DICT = {element: index for index, element in enumerate(ELEMENT_ORDER)}


def parse_formula(formula: str) -> tuple[dict[str, int], int]:
    def parse_segment(segment: str):
        element_counts = defaultdict(int)
        pattern = r"([A-Z][a-z]?)(\d*)|\(([^()]+)\)(\d*)"
        for match in re.finditer(pattern, segment):
            if match.group(1):
                element = match.group(1)
                count = int(match.group(2)) if match.group(2) else 1
                element_counts[element] += count
            elif match.group(3):
                inner = match.group(3)
                multiplier = int(match.group(4)) if match.group(4) else 1
                inner_counts = parse_segment(inner)
                for element, count in inner_counts.items():
                    element_counts[element] += count * multiplier
        return element_counts

    element_counts = parse_segment(formula)
    total_atoms_in_formula_unit = sum(element_counts.values())
    sorted_elements = sorted(
        element_counts.items(),
        key=lambda item: ELEMENT_ORDER_DICT.get(item[0], float("inf")),
    )
    sorted_element_counts = OrderedDict(sorted_elements)
    return dict(sorted_element_counts), total_atoms_in_formula_unit


def build_formula_vocab(formulas: Iterable[str]) -> Dict[str, int]:
    elements = set()
    for formula in formulas:
        element_counts, _ = parse_formula(formula)
        elements.update(element_counts.keys())
    element_list = sorted(list(elements))
    element_to_idx = {e: idx + 2 for idx, e in enumerate(element_list)}
    element_to_idx["<pad>"] = 0
    element_to_idx["<unk>"] = 1
    return element_to_idx


def formula_to_tokens(
    formula: str,
    n_atoms: int,
    formula_vocab: Dict[str, int],
    max_atoms: int,
) -> tuple[list[int], list[int]]:
    element_counts, total_atoms_in_formula_unit = parse_formula(formula)

    element_total_atoms = {}
    total_floor_counts = 0
    fractional_parts = []

    for element, count in element_counts.items():
        total_atoms = n_atoms * count / total_atoms_in_formula_unit
        floor_count = int(total_atoms)
        fractional_part = total_atoms - floor_count
        element_total_atoms[element] = floor_count
        total_floor_counts += floor_count
        fractional_parts.append((fractional_part, element))

    missing_atoms = n_atoms - total_floor_counts
    fractional_parts.sort(reverse=True)

    for i in range(missing_atoms):
        _, element = fractional_parts[i]
        element_total_atoms[element] += 1

    tokens = []
    for element, count in element_total_atoms.items():
        element_idx = formula_vocab.get(element, formula_vocab["<unk>"])
        tokens.extend([element_idx] * count)

    if len(tokens) > max_atoms:
        tokens = tokens[:max_atoms]
        mask = [1] * max_atoms
    else:
        mask = [1] * len(tokens) + [0] * (max_atoms - len(tokens))
        tokens += [formula_vocab["<pad>"]] * (max_atoms - len(tokens))

    return tokens, mask