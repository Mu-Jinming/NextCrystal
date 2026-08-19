#!/usr/bin/env python3
"""Prepare NextDiff CIF outputs for an external evaluator.

NextDiff writes structures as ``1.cif``, ``2.cif``, ... in JSON-query order.
The composition-level evaluation wrapper uses the explicit filename schema
``n_i_formula_atoms.cif``.  This script reconstructs that mapping from the
post-processing CSV and the exact JSON file used for sampling, verifies their
order, and creates a lightweight evaluation directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate.convert_format_json import (  # noqa: E402
    compress_to_tokens,
    load_wy_tokens,
    parse_assignment_string,
    parse_list_field,
    validate_scheme,
)


def _parse_nonnegative_integer(value: str) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    integer = int(number)
    return integer if number == integer and integer >= 0 else None


def assign_input_numbers(rows: list[dict[str, str]]) -> dict[str, int]:
    """Map source ``cif_name`` values to one-based evaluation indices."""
    ordered_ids = list(dict.fromkeys((row.get("cif_name") or "").strip() for row in rows))
    if not ordered_ids or any(not item for item in ordered_ids):
        raise ValueError("Every assignment row must contain a non-empty cif_name")

    parsed = [_parse_nonnegative_integer(item) for item in ordered_ids]
    if all(value is not None for value in parsed) and len(set(parsed)) == len(parsed):
        return {item: int(value) + 1 for item, value in zip(ordered_ids, parsed)}
    return {item: index for index, item in enumerate(ordered_ids, start=1)}


def sanitize_formula(value: str) -> str:
    formula = re.sub(r"\s+", "", value or "")
    formula = re.sub(r"[^A-Za-z0-9()+.\-]", "-", formula)
    if not formula:
        raise ValueError("Formula pretty is empty after filename sanitization")
    return formula


def _query_signature(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "spacegroup_number": int(record["spacegroup_number"]),
        "wyckoff_letters": list(record["wyckoff_letters"]),
        "atom_types": list(record["atom_types"]),
    }


def build_manifest(
    assignment_csv: Path,
    query_json: Path,
    cif_dir: Path,
    wy_tokens_path: Path,
) -> list[dict[str, Any]]:
    """Reconstruct one manifest row for every valid sampled query."""
    with assignment_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with query_json.open("r", encoding="utf-8") as handle:
        query_records = json.load(handle)
    if not isinstance(query_records, list):
        raise ValueError("The NextDiff query JSON must contain a list")

    input_numbers = assign_input_numbers(rows)
    wy_tokens = load_wy_tokens(str(wy_tokens_path))
    candidate_ranks: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    query_offset = 0

    for row_number, row in enumerate(rows, start=2):
        input_id = (row.get("cif_name") or "").strip()
        sg_raw = (row.get("Spacegroup Number") or "").strip()
        natoms_raw = (row.get("NAtoms") or "").strip()
        if not sg_raw or not natoms_raw:
            continue

        sg_num = int(float(sg_raw))
        natoms = int(float(natoms_raw))
        formula = sanitize_formula(
            row.get("Formula pretty") or row.get("pretty_formula") or ""
        )

        try:
            schemes = parse_list_field(row.get("Assignments", ""))
        except Exception:
            continue

        for assignment in schemes:
            try:
                expanded = parse_assignment_string(str(assignment))
                atom_types, tokens, multiplicities = compress_to_tokens(
                    expanded, sg_num, wy_tokens
                )
                is_valid, _ = validate_scheme(
                    sg_num, natoms, tokens, multiplicities, wy_tokens
                )
            except Exception:
                continue
            if not is_valid:
                continue

            expected = {
                "spacegroup_number": sg_num,
                "wyckoff_letters": tokens,
                "atom_types": atom_types,
            }
            if query_offset >= len(query_records):
                raise ValueError(
                    "Assignment CSV contains more valid candidates than the query JSON"
                )
            actual = _query_signature(query_records[query_offset])
            if actual != expected:
                raise ValueError(
                    "CSV/JSON query-order mismatch at one-based query "
                    f"{query_offset + 1} (CSV row {row_number})"
                )

            candidate_ranks[input_id] += 1
            query_index = query_offset + 1
            input_number = input_numbers[input_id]
            rank = candidate_ranks[input_id]
            filename = f"{input_number}_{rank}_{formula}_{natoms}.cif"
            manifest.append(
                {
                    "query_index": query_index,
                    "input_number": input_number,
                    "source_input_id": input_id,
                    "candidate_rank": rank,
                    "formula": formula,
                    "num_atoms": natoms,
                    "spacegroup_number": sg_num,
                    "source_cif": str((cif_dir / f"{query_index}.cif").resolve()),
                    "evaluation_filename": filename,
                }
            )
            query_offset += 1

    if query_offset != len(query_records):
        raise ValueError(
            "Query JSON contains more records than could be reconstructed from the "
            f"assignment CSV ({len(query_records)} != {query_offset})"
        )
    return manifest


def materialize_manifest(
    manifest: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_cifs = list(output_dir.glob("*.cif"))
    if existing_cifs:
        raise FileExistsError(
            f"Output directory already contains {len(existing_cifs)} CIF files: "
            f"{output_dir}"
        )

    missing = [Path(row["source_cif"]) for row in manifest if not Path(row["source_cif"]).is_file()]
    if missing and not allow_missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} sampled CIF files; first paths: {preview}"
        )

    written: list[dict[str, Any]] = []
    for row in manifest:
        source = Path(row["source_cif"])
        destination = output_dir / row["evaluation_filename"]
        record = dict(row)
        record["available"] = source.is_file()
        record["evaluation_cif"] = str(destination.resolve())
        if source.is_file():
            if mode == "symlink":
                destination.symlink_to(os.path.relpath(source, destination.parent))
            elif mode == "hardlink":
                os.link(source, destination)
            elif mode == "copy":
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unsupported materialization mode: {mode}")
        written.append(record)
    return written


def write_manifest(records: list[dict[str, Any]], output_dir: Path) -> None:
    json_path = output_dir / "manifest.json"
    csv_path = output_dir / "manifest.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    fieldnames = list(records[0]) if records else ["query_index"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-csv", required=True, type=Path)
    parser.add_argument("--query-json", required=True, type=Path)
    parser.add_argument("--cif-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--wy-tokens",
        type=Path,
        default=PROJECT_ROOT / "generate" / "wy_tokens_complete.json",
    )
    parser.add_argument(
        "--mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        args.assignment_csv.resolve(),
        args.query_json.resolve(),
        args.cif_dir.resolve(),
        args.wy_tokens.resolve(),
    )
    records = materialize_manifest(
        manifest, args.output_dir.resolve(), args.mode, args.allow_missing
    )
    write_manifest(records, args.output_dir.resolve())

    available = sum(bool(record["available"]) for record in records)
    inputs = len({record["input_number"] for record in records})
    print(
        f"Prepared {available}/{len(records)} candidate CIFs for {inputs} inputs "
        f"under {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
