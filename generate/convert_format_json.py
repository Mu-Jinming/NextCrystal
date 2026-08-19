#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

WY_TOKEN_PATH_DEFAULT = "./generate/wy_tokens_complete.json"


def load_wy_tokens(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("wy_tokens_complete.json must be a JSON object/dict")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = [str(v)]
    return out


def parse_list_field(raw: str) -> List[Any]:
    """
    Parse Assignments cell which stores a JSON list, e.g.
      '["O:g, O:g; Hf:f", "O:b, O:b; Hf:f", ...]'
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, list):
            return obj
    except Exception as e:
        raise ValueError(f"Cannot parse Assignments list: {raw[:120]}...") from e
    raise ValueError(f"Unexpected structure for Assignments list: {type(obj)}")


def parse_assignment_string(assignment: str) -> List[Tuple[str, str]]:
    """
    Example:
      "O:f, O:f, O:f, O:f; Hf:a, Hf:a"
    Return expanded pairs:
      [("O","f"), ("O","f"), ("O","f"), ("O","f"), ("Hf","a"), ("Hf","a")]
    """
    assignment = (assignment or "").strip()
    if not assignment:
        return []

    parts = [p.strip() for p in assignment.split(";") if p.strip()]
    expanded: List[Tuple[str, str]] = []
    for part in parts:
        items = [x.strip() for x in part.split(",") if x.strip()]
        for item in items:
            if ":" not in item:
                raise ValueError(f"Bad assignment item (missing ':'): {item}")
            atom, letter = item.split(":", 1)
            atom = atom.strip()
            letter = letter.strip()
            if not atom or not letter:
                raise ValueError(f"Bad assignment item: {item}")
            expanded.append((atom, letter))
    return expanded


def _build_letter_mult_map(sg_num: int, wy_tokens: Dict[str, List[str]]) -> Dict[str, int]:
    """
    From wy_tokens (e.g. ["2a","2b","6c","12d"]), build letter->multiplicity mapping.
    Assumption: each letter has a unique multiplicity in a given space group.
    """
    sg_key = f"SG_{sg_num}"
    allowed = wy_tokens.get(sg_key, [])
    mult_map: Dict[str, int] = {}
    for t in allowed:
        m = re.match(r"^(\d+)([A-Za-z])$", str(t).strip())
        if not m:
            continue
        mult = int(m.group(1))
        # Space group 47 has both the general position ``8A`` and the fixed
        # position ``1a``.  Wyckoff letters are therefore case-sensitive in
        # this token table; lowercasing merges two distinct positions.
        letter = m.group(2)
        mult_map.setdefault(letter, mult)
    return mult_map


def compress_to_tokens(
    expanded: List[Tuple[str, str]],
    sg_num: int,
    wy_tokens: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[int]]:
    """
    Example outputs:
      atom_types: ["O","O","Hf",...]
      tokens:     ["2b","2b","2b",...]
      mults:      [2, 2, 2, ...]
    """
    if not expanded:
        return [], [], []

    mult_map = _build_letter_mult_map(sg_num, wy_tokens)
    if not mult_map:
        return [], [], []

    counts: Dict[Tuple[str, str], int] = {}
    first_pos: Dict[Tuple[str, str], int] = {}

    for i, (atom, letter) in enumerate(expanded):
        key = (atom, letter)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_pos:
            first_pos[key] = i

    groups = sorted(counts.keys(), key=lambda k: first_pos.get(k, 10**9))

    atom_types: List[str] = []
    tokens: List[str] = []
    mults: List[int] = []

    for atom, letter in groups:
        if letter not in mult_map:
            raise ValueError(f"Letter '{letter}' not in wy_tokens for SG_{sg_num}")

        mult = mult_map[letter]
        cnt = counts[(atom, letter)]

        if mult <= 0:
            raise ValueError(f"Bad multiplicity for letter '{letter}' in SG_{sg_num}: {mult}")

        if cnt % mult != 0:
            raise ValueError(
                f"Count {cnt} of {atom}:{letter} not divisible by multiplicity {mult} in SG_{sg_num}"
            )

        n_orbits = cnt // mult
        for _ in range(n_orbits):
            atom_types.append(atom)
            tokens.append(f"{mult}{letter}")
            mults.append(mult)

    return atom_types, tokens, mults


def is_valid_scheme(
    sg_num: int,
    natoms: int,
    tokens: List[str],
    mults: List[int],
    wy_tokens: Dict[str, List[str]],
) -> bool:
    sg_key = f"SG_{sg_num}"
    allowed = set(wy_tokens.get(sg_key, []))
    if not allowed:
        return False

    if not tokens or not all(t in allowed for t in tokens):
        return False

    if sum(mults) != natoms:
        return False

    return True


from collections import Counter

def validate_scheme(sg_num, natoms, tokens, mults, wy_tokens):
    sg_key = f"SG_{sg_num}"
    allowed = set(wy_tokens.get(sg_key, []))
    if not allowed:
        return False, "sg_missing_in_wy_tokens"

    if not tokens:
        return False, "empty_tokens"

    bad = [t for t in tokens if t not in allowed]
    if bad:
        return False, f"token_not_allowed: {bad[:5]}"

    s = sum(mults)
    if s != natoms:
        return False, f"natoms_mismatch: sum(mults)={s} natoms={natoms}"

    return True, "ok"


def convert_with_diagnostics(csv_path: str, wy_token_path: str):
    wy_tokens = load_wy_tokens(wy_token_path)

    total_schemes = 0
    valid_schemes = 0
    invalid_schemes = 0

    reason_counter = Counter()
    invalid_samples = []

    output = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            sg_num_raw = (row.get("Spacegroup Number") or "").strip()
            natoms_raw = (row.get("NAtoms") or "").strip()
            assignments_raw = row.get("Assignments", "")

            if not sg_num_raw or not natoms_raw:
                continue

            sg_num = int(float(sg_num_raw))
            natoms = int(float(natoms_raw))

            try:
                schemes = parse_list_field(assignments_raw)
            except Exception:
                continue

            for cand in schemes:
                total_schemes += 1
                cand_str = str(cand)

                try:
                    expanded = parse_assignment_string(cand_str)
                    atom_types, tokens, mults = compress_to_tokens(expanded, sg_num, wy_tokens)

                    ok, reason = validate_scheme(sg_num, natoms, tokens, mults, wy_tokens)
                    if ok:
                        valid_schemes += 1
                        output.append(
                            {
                                "spacegroup_number": sg_num,
                                "wyckoff_letters": tokens,
                                "atom_types": atom_types,
                            }
                        )
                    else:
                        invalid_schemes += 1
                        reason_counter[reason] += 1
                        invalid_samples.append({
                            "sg": sg_num,
                            "natoms": natoms,
                            "cand": cand_str,
                            "tokens": tokens,
                            "mults": mults,
                            "reason": reason
                        })

                except Exception as e:
                    invalid_schemes += 1
                    reason = f"parse_exception: {type(e).__name__}: {str(e)[:120]}"
                    reason_counter[reason] += 1
                    invalid_samples.append({
                        "sg": sg_num,
                        "natoms": natoms,
                        "cand": cand_str,
                        "reason": reason
                    })

    stats = {
        "total_schemes": total_schemes,
        "valid_schemes": valid_schemes,
        "invalid_schemes": invalid_schemes,
        "invalid_reasons": dict(reason_counter),
    }
    return output, stats, invalid_samples


def convert(csv_path: str, wy_token_path: str):
    """Convert assignments while preserving the original two-value API."""
    output, stats, _invalid_samples = convert_with_diagnostics(
        csv_path, wy_token_path
    )
    return output, stats


def main():
    ap = argparse.ArgumentParser(
        description="Convert CSV to JSON: keep ALL valid schemes only "
                    "(tokens in wy_tokens AND multiplicity sum == NAtoms)."
    )
    ap.add_argument("input_csv", help="input postprocessed_assignments CSV file path")
    ap.add_argument("output_json", help="output JSON file path")
    ap.add_argument(
        "--wy_tokens",
        default=WY_TOKEN_PATH_DEFAULT,
        help=f"path to wy_tokens_complete.json (default: {WY_TOKEN_PATH_DEFAULT})",
    )
    ap.add_argument("--indent", type=int, default=4, help="JSON indent (default: 4)")
    ap.add_argument(
        "--invalid-debug-output",
        help=(
            "optional invalid-scheme report path; by default a report is written "
            "next to output_json only when invalid schemes exist"
        ),
    )
    args = ap.parse_args()

    records, stats, invalid_samples = convert_with_diagnostics(
        args.input_csv, args.wy_tokens
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=args.indent)

    invalid_debug_path = None
    if invalid_samples:
        output_path = Path(args.output_json)
        invalid_debug_path = (
            Path(args.invalid_debug_output)
            if args.invalid_debug_output
            else output_path.with_suffix(".invalid_debug.json")
        )
        invalid_debug_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_debug_path.write_text(
            json.dumps(invalid_samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("=== Scheme Statistics ===")
    print(f"Total schemes   : {stats['total_schemes']}")
    print(f"Valid schemes   : {stats['valid_schemes']}")
    print(f"Invalid schemes : {stats['invalid_schemes']}")
    if invalid_debug_path is not None:
        print(f"Invalid details : {invalid_debug_path}")
    print(f"Done. Wrote {len(records)} records to {args.output_json}")


if __name__ == "__main__":
    main()
