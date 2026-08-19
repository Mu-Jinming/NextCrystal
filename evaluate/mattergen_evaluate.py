#!/usr/bin/env python3
"""Composition-level best-of-K wrapper around external ``mattergen-evaluate``.

This file contains no MatterGen implementation.  It validates and invokes the
CLI installed by a separate MatterGen environment, evaluates candidates in
ascending rank, and reports composition-level Stability, Uniqueness, Novelty,
and SUN.  Uniqueness follows the paper's within-composition convention: after
sorting one composition's candidates by rank, the first candidate is the first
representative of its local duplicate set.  Consequently, a composition with
at least one candidate in the requested prefix is Unique at that K.  This is
deliberately different from global cross-composition pool uniqueness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent
    / "reference"
    / "reference_MP2020correction.gz"
)
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_HEADER)) == LFS_HEADER


def require_materialized_reference(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference dataset not found: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            "The MP2020 reference is still the repository's 134-byte pointer "
            "placeholder. Download the external 873,410,170-byte "
            "reference_MP2020correction.gz payload and replace this file "
            "before evaluation."
        )
    return path


def resolve_executable(command: str) -> str:
    if os.sep in command:
        path = Path(command).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"MatterGen evaluator is not executable: {path}")
        return str(path)
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Cannot find `{command}`. Install MatterGen in a separate environment "
            "or pass --mattergen-command."
        )
    return resolved


def metric_value(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def is_one(value: float) -> bool:
    return value > 0.999999


def parse_evaluation_cif_name(cif_path: Path) -> tuple[int, int, str, int]:
    parts = cif_path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            "Bad CIF name; expected n_i_formula_atoms.cif: " f"{cif_path.name}"
        )
    input_number = int(parts[0])
    candidate_rank = int(parts[1])
    num_atoms = int(parts[-1])
    formula = "_".join(parts[2:-1])
    if not formula:
        raise ValueError(f"Empty formula in CIF name: {cif_path.name}")
    return input_number, candidate_rank, formula, num_atoms


def composition_unique_representative(
    candidates: list[tuple[int, str]],
) -> dict[str, Any] | None:
    """Return the first local representative used by composition-level U@K.

    The archived response analysis applies structural deduplication separately
    inside each official composition and then computes ``any(unique)`` over the
    rank prefix.  Whatever later candidates match, the first candidate in a
    non-empty local pool is necessarily a unique representative.  Therefore
    the composition-level endpoint can be evaluated without re-running an
    all-pairs matcher.  Missing compositions remain unsuccessful rather than
    being hard-coded as unique.
    """
    if not candidates:
        return None
    rank, path_string = min(candidates, key=lambda item: item[0])
    path = Path(path_string)
    return {
        "filename": path.name,
        "path": str(path),
        "candidate_rank": rank,
        "scope": "within_official_composition_index",
    }


def build_command(
    executable: str,
    structures_dir: Path,
    metrics_path: Path,
    reference_dataset: Path,
    relax: bool,
    structure_matcher: str,
    device: str | None,
    potential_load_path: str | None,
    save_relaxed: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        executable,
        f"--structures_path={structures_dir}",
        f"--relax={'True' if relax else 'False'}",
        f"--structure_matcher={structure_matcher}",
        f"--save_as={metrics_path}",
        f"--reference_dataset_path={reference_dataset}",
        "--energy_correction_scheme=MP2020",
    ]
    if device:
        command.append(f"--device={device}")
    if potential_load_path:
        command.append(f"--potential_load_path={potential_load_path}")
    if save_relaxed:
        command.append(
            f"--structures_output_path={metrics_path.parent / 'relaxed.extxyz'}"
        )
    return command + list(extra_args)


def evaluate_one_candidate(
    cif_path: Path,
    out_dir: Path,
    executable: str,
    reference_dataset: Path,
    relax: bool,
    structure_matcher: str,
    device: str | None,
    potential_load_path: str | None,
    save_relaxed: bool,
    extra_args: list[str],
) -> tuple[bool, dict[str, Any] | None, str | None]:
    work_dir = out_dir / "_work" / cif_path.stem
    structures_dir = work_dir / "structures"
    metrics_path = work_dir / "metrics.json"
    log_path = work_dir / "mattergen-evaluate.log"
    try:
        structures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cif_path, structures_dir / cif_path.name)
        command = build_command(
            executable,
            structures_dir,
            metrics_path,
            reference_dataset,
            relax,
            structure_matcher,
            device,
            potential_load_path,
            save_relaxed,
            extra_args,
        )
        process = subprocess.run(
            command,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(process.stdout, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(
                f"mattergen-evaluate exited with {process.returncode}; see {log_path}"
            )
        if not metrics_path.is_file():
            raise RuntimeError(f"MatterGen did not write {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        stable = is_one(metric_value(metrics, "frac_stable_structures"))
        novel = is_one(metric_value(metrics, "frac_novel_structures"))
        return (
            True,
            {
                "filename": cif_path.name,
                "path": str(cif_path),
                "stable": stable,
                "novel": novel,
                "stable_novel": stable and novel,
                "metrics": metrics,
            },
            None,
        )
    except Exception:
        return False, None, traceback.format_exc()


def evaluate_one_input(
    input_number: int,
    candidates: list[tuple[int, str]],
    formula: str,
    num_atoms: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    unique_representative = composition_unique_representative(candidates)
    representatives: dict[str, dict[str, Any] | None] = {
        "stable": None,
        "novel": None,
        "stable_novel": None,
    }
    errors: list[dict[str, Any]] = []
    num_ok = 0

    for rank, path_string in candidates:
        ok, result, error = evaluate_one_candidate(
            Path(path_string),
            Path(options["out_dir"]),
            options["executable"],
            Path(options["reference_dataset"]),
            options["relax"],
            options["structure_matcher"],
            options["device"],
            options["potential_load_path"],
            options["save_relaxed"],
            options["extra_args"],
        )
        if not ok:
            errors.append({"rank": rank, "path": path_string, "error": error})
            continue
        num_ok += 1
        assert result is not None
        result["candidate_rank"] = rank
        if result["stable"] and representatives["stable"] is None:
            representatives["stable"] = dict(result)
        if result["novel"] and representatives["novel"] is None:
            representatives["novel"] = dict(result)
        if result["stable_novel"]:
            representatives["stable_novel"] = dict(result)
            representatives["stable"] = dict(result)
            representatives["novel"] = dict(result)
            break

    record = {
        "n": input_number,
        "pretty_formula": formula,
        "atoms": num_atoms,
        "num_cifs": len(candidates),
        "num_ok": num_ok,
        "num_fail": len(errors),
        "stable": representatives["stable"] is not None,
        "unique": unique_representative is not None,
        "novel": representatives["novel"] is not None,
        "stable_novel": representatives["stable_novel"] is not None,
        "sun": (
            unique_representative is not None
            and representatives["stable_novel"] is not None
        ),
        "rep_stable": representatives["stable"],
        "rep_unique": unique_representative,
        "rep_novel": representatives["novel"],
        "rep_stable_novel": representatives["stable_novel"],
    }
    return {"record": record, "errors": errors}


def _representative_metric(
    representative: dict[str, Any] | None, key: str
) -> str:
    if not representative:
        return ""
    value = metric_value(representative.get("metrics", {}), key, float("nan"))
    return "" if value != value else str(value)


def write_outputs(
    records: dict[int, dict[str, Any]],
    errors: list[dict[str, Any]],
    out_dir: Path,
    num_inputs: int,
    candidate_budget: int,
) -> None:
    summary_jsonl = out_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as handle:
        for input_number in range(1, num_inputs + 1):
            record = records[input_number]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    columns = [
        "n",
        "pretty_formula",
        "atoms",
        "num_cifs",
        "num_ok",
        "num_fail",
        "stable",
        "unique",
        "novel",
        "stable_novel",
        "sun",
        "rep_stable",
        "rep_unique",
        "rep_novel",
        "rep_stable_novel",
        "stable_energy_above_hull_per_atom",
        "novel_energy_above_hull_per_atom",
        "stable_novel_energy_above_hull_per_atom",
    ]
    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for input_number in range(1, num_inputs + 1):
            record = records[input_number]
            stable = record.get("rep_stable")
            unique = record.get("rep_unique")
            novel = record.get("rep_novel")
            stable_novel = record.get("rep_stable_novel")
            writer.writerow(
                {
                    "n": input_number,
                    "pretty_formula": record.get("pretty_formula", ""),
                    "atoms": record.get("atoms", ""),
                    "num_cifs": record.get("num_cifs", 0),
                    "num_ok": record.get("num_ok", 0),
                    "num_fail": record.get("num_fail", 0),
                    "stable": int(bool(record.get("stable"))),
                    "unique": int(bool(record.get("unique"))),
                    "novel": int(bool(record.get("novel"))),
                    "stable_novel": int(bool(record.get("stable_novel"))),
                    "sun": int(bool(record.get("sun"))),
                    "rep_stable": stable.get("filename", "") if stable else "",
                    "rep_unique": unique.get("filename", "") if unique else "",
                    "rep_novel": novel.get("filename", "") if novel else "",
                    "rep_stable_novel": (
                        stable_novel.get("filename", "") if stable_novel else ""
                    ),
                    "stable_energy_above_hull_per_atom": _representative_metric(
                        stable, "avg_energy_above_hull_per_atom"
                    ),
                    "novel_energy_above_hull_per_atom": _representative_metric(
                        novel, "avg_energy_above_hull_per_atom"
                    ),
                    "stable_novel_energy_above_hull_per_atom": (
                        _representative_metric(
                            stable_novel, "avg_energy_above_hull_per_atom"
                        )
                    ),
                }
            )

    numerators = {
        key: sum(bool(record.get(key)) for record in records.values())
        for key in ("stable", "unique", "novel", "sun", "stable_novel")
    }
    aggregate = {
        "candidate_budget": candidate_budget,
        "denominator": num_inputs,
        "numerators": numerators,
        "rates": {key: value / num_inputs for key, value in numerators.items()},
        "metric_definitions": {
            "stable": "any stable candidate with rank <= K",
            "unique": (
                "any within-composition unique representative with rank <= K; "
                "the first available ranked candidate is necessarily unique"
            ),
            "novel": "any novel candidate with rank <= K",
            "sun": "any single candidate with rank <= K that is stable and novel, and a non-empty local candidate prefix",
            "denominator": "complete official test split; an absent composition is unsuccessful",
        },
        "uniqueness_scope": "within_official_composition_index",
        "candidate_failures": sum(record.get("num_fail", 0) for record in records.values()),
        "errors": errors,
    }
    with (out_dir / "aggregate_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-inputs", required=True, type=int)
    parser.add_argument(
        "--candidate-budget",
        type=int,
        choices=(1, 5, 10, 20),
        default=20,
        help="Evaluate the nested rank prefix K (default: 20).",
    )
    parser.add_argument(
        "--reference-dataset", type=Path, default=DEFAULT_REFERENCE
    )
    parser.add_argument("--mattergen-command", default="mattergen-evaluate")
    parser.add_argument("--relax", choices=("True", "False"), default="True")
    parser.add_argument(
        "--structure-matcher", choices=("ordered", "disordered"), default="disordered"
    )
    parser.add_argument("--device")
    parser.add_argument("--potential-load-path")
    parser.add_argument("--save-relaxed", action="store_true")
    parser.add_argument("--copy-cifs", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    reference_dataset = require_materialized_reference(args.reference_dataset)
    executable = resolve_executable(args.mattergen_command)
    if args.check_only:
        print(f"mattergen-evaluate: {executable}")
        print(f"reference dataset: {reference_dataset}")
        return

    if args.num_inputs <= 0:
        raise ValueError("--num-inputs must be positive")
    cif_dir = args.cif_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (out_dir / "summary.jsonl", out_dir / "summary.csv"):
        if path.exists():
            raise FileExistsError(
                f"Refusing to mix a new run with existing output: {path}"
            )

    groups: dict[int, list[tuple[int, str]]] = {}
    metadata: dict[int, tuple[str, int]] = {}
    invalid_names: list[str] = []
    for cif_path in sorted(cif_dir.rglob("*.cif")):
        try:
            input_number, rank, formula, num_atoms = parse_evaluation_cif_name(
                cif_path
            )
        except Exception:
            invalid_names.append(str(cif_path))
            continue
        if not 1 <= input_number <= args.num_inputs:
            continue
        if rank > args.candidate_budget:
            continue
        groups.setdefault(input_number, []).append((rank, str(cif_path)))
        metadata.setdefault(input_number, (formula, num_atoms))
    if not groups:
        raise ValueError(f"No evaluation-formatted CIF files found under {cif_dir}")
    for candidates in groups.values():
        candidates.sort(key=lambda item: item[0])

    options = {
        "out_dir": str(out_dir),
        "executable": executable,
        "reference_dataset": str(reference_dataset),
        "relax": args.relax == "True",
        "structure_matcher": args.structure_matcher,
        "device": args.device,
        "potential_load_path": args.potential_load_path,
        "save_relaxed": args.save_relaxed,
        "extra_args": args.extra,
    }
    records: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        for input_number in range(1, args.num_inputs + 1):
            candidates = groups.get(input_number, [])
            formula, num_atoms = metadata.get(input_number, ("", 0))
            if not candidates:
                records[input_number] = {
                    "n": input_number,
                    "pretty_formula": formula,
                    "atoms": num_atoms,
                    "num_cifs": 0,
                    "num_ok": 0,
                    "num_fail": 0,
                    "stable": False,
                    "unique": False,
                    "novel": False,
                    "stable_novel": False,
                    "sun": False,
                    "rep_stable": None,
                    "rep_unique": None,
                    "rep_novel": None,
                    "rep_stable_novel": None,
                }
                continue
            futures.append(
                executor.submit(
                    evaluate_one_input,
                    input_number,
                    candidates,
                    formula,
                    num_atoms,
                    options,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            record = result["record"]
            records[int(record["n"])] = record
            errors.extend(result["errors"])

    if args.copy_cifs:
        for label, key in (
            ("stable", "rep_stable"),
            ("novel", "rep_novel"),
            ("stable_novel", "rep_stable_novel"),
        ):
            destination = out_dir / label
            destination.mkdir(exist_ok=True)
            for record in records.values():
                representative = record.get(key)
                if representative:
                    source = Path(representative["path"])
                    shutil.copy2(source, destination / source.name)

    if invalid_names:
        errors.append({"invalid_cif_names": invalid_names})
    write_outputs(
        records,
        errors,
        out_dir,
        args.num_inputs,
        args.candidate_budget,
    )
    print(f"Wrote MatterGen evaluation summaries under {out_dir}")


if __name__ == "__main__":
    main()
