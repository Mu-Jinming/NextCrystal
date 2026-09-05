#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Protocol


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent
    / "reference"
    / "reference_MP2020correction.gz"
)
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class StructureMatcherProtocol(Protocol):
    def fit(self, structure_a: Any, structure_b: Any) -> bool: ...


def make_structure_matcher(kind: str) -> StructureMatcherProtocol:
    try:
        from mattergen.evaluation.utils.structure_matcher import (
            DefaultDisorderedStructureMatcher,
            DefaultOrderedStructureMatcher,
        )
    except ImportError as error:
        raise RuntimeError(
            "SUN uniqueness filtering requires the MatterGen Python package "
            "in the wrapper environment so it can use the same StructureMatcher "
            "implementation as mattergen-evaluate."
        ) from error

    if kind == "ordered":
        return DefaultOrderedStructureMatcher()
    if kind == "disordered":
        return DefaultDisorderedStructureMatcher()
    raise ValueError(f"Unknown structure matcher: {kind!r}")


def load_evaluated_structure(path: str | Path) -> Any:
    structure_path = Path(path)
    if not structure_path.is_file():
        raise FileNotFoundError(
            f"Structure for SUN matching not found: {structure_path}"
        )

    if structure_path.suffix.lower() in {".extxyz", ".xyz"}:
        try:
            from ase.io import read
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as error:
            raise RuntimeError(
                "Reading relaxed EXTXYZ structures requires ASE and pymatgen."
            ) from error
        atoms = read(structure_path, index=-1, format="extxyz")
        return AseAtomsAdaptor.get_structure(atoms)

    try:
        from pymatgen.core import Structure
    except ImportError as error:
        raise RuntimeError("Reading CIF structures requires pymatgen.") from error
    return Structure.from_file(structure_path)


def filter_unique_candidates(
    candidates: list[dict[str, Any]],
    matcher_kind: str,
    *,
    matcher: StructureMatcherProtocol | None = None,
    structure_loader: Callable[[str | Path], Any] = load_evaluated_structure,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["candidate_rank"]),
            int(item.get("input_number", 0)),
            str(item["filename"]),
        ),
    )

    active_matcher = matcher or make_structure_matcher(matcher_kind)
    annotated: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    representatives_by_composition: dict[str, list[dict[str, Any]]] = {}
    structures_by_composition: dict[str, list[Any]] = {}

    for candidate in ordered:
        result = dict(candidate)
        structure = structure_loader(result["match_structure_path"])
        composition = getattr(structure, "composition", None)
        composition_key = (
            str(composition.reduced_formula)
            if composition is not None
            else "__all_candidates__"
        )
        local_representatives = representatives_by_composition.setdefault(
            composition_key, []
        )
        local_structures = structures_by_composition.setdefault(
            composition_key, []
        )
        duplicate_index: int | None = None
        for index, representative_structure in enumerate(local_structures):
            if active_matcher.fit(structure, representative_structure):
                duplicate_index = index
                break

        if duplicate_index is None:
            result["unique"] = True
            result["duplicate_of_input_number"] = None
            result["duplicate_of_candidate_rank"] = None
            result["duplicate_of_filename"] = None
            representatives.append(result)
            local_representatives.append(result)
            local_structures.append(structure)
        else:
            representative = local_representatives[duplicate_index]
            result["unique"] = False
            result["duplicate_of_input_number"] = representative.get(
                "input_number"
            )
            result["duplicate_of_candidate_rank"] = int(
                representative["candidate_rank"]
            )
            result["duplicate_of_filename"] = representative["filename"]
        annotated.append(result)

    return annotated, representatives


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
            "The MP2020 reference is still the repository's 134-byte Git LFS pointer "
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


def metric_value(
    metrics: dict[str, Any], key: str, default: float | None = None
) -> float:
    if key not in metrics:
        if default is None:
            raise KeyError(f"MatterGen metrics are missing required field: {key}")
        return float(default)
    value = metrics[key]
    if isinstance(value, dict):
        if "value" not in value:
            if default is None:
                raise KeyError(
                    f"MatterGen metric {key!r} is missing its value field"
                )
            return float(default)
        value = value["value"]
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        if default is None:
            raise ValueError(
                f"MatterGen metric {key!r} is not numeric: {value!r}"
            ) from error
        return float(default)
    if not math.isfinite(result):
        if default is None:
            raise ValueError(f"MatterGen metric {key!r} is not finite: {result}")
        return float(default)
    return result


def is_one(value: float) -> bool:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Expected a fraction in [0, 1], received {value}")
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


def composition_unique_candidate(
    candidates: list[tuple[int, str]],
) -> dict[str, Any] | None:
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
    if relax:
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
    extra_args: list[str],
) -> tuple[bool, dict[str, Any] | None, str | None]:
    work_dir = out_dir / "_work" / cif_path.stem
    structures_dir = work_dir / "structures"
    metrics_path = work_dir / "metrics.json"
    log_path = work_dir / "mattergen-evaluate.log"
    try:
        structures_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)
        (work_dir / "relaxed.extxyz").unlink(missing_ok=True)
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
        if relax:
            match_structure_path = work_dir / "relaxed.extxyz"
            if not match_structure_path.is_file():
                raise RuntimeError(
                    "MatterGen did not serialize the relaxed structure required "
                    f"for SUN uniqueness matching: {match_structure_path}"
                )
            structure_stage = "mattergen_relaxed"
        else:
            match_structure_path = cif_path
            structure_stage = "input_cif"
        return (
            True,
            {
                "filename": cif_path.name,
                "path": str(cif_path),
                "stable": stable,
                "novel": novel,
                "stable_novel": stable and novel,
                "match_structure_path": str(match_structure_path),
                "match_structure_stage": structure_stage,
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
    candidates = sorted(candidates, key=lambda item: (item[0], item[1]))
    unique_candidate = composition_unique_candidate(candidates)
    representatives: dict[str, dict[str, Any] | None] = {
        "stable": None,
        "novel": None,
        "stable_novel": None,
        "stable_unique_novel": None,
    }
    evaluated_candidates: list[dict[str, Any]] = []
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
            options["extra_args"],
        )
        if not ok:
            errors.append({"rank": rank, "path": path_string, "error": error})
            continue
        num_ok += 1
        assert result is not None
        result["candidate_rank"] = rank
        evaluated_candidates.append(dict(result))
        if result["stable"] and representatives["stable"] is None:
            representatives["stable"] = dict(result)
        if result["novel"] and representatives["novel"] is None:
            representatives["novel"] = dict(result)
        if result["stable_novel"]:
            if representatives["stable_novel"] is None:
                representatives["stable_novel"] = dict(result)
    record = {
        "n": input_number,
        "pretty_formula": formula,
        "atoms": num_atoms,
        "num_cifs": len(candidates),
        "num_ok": num_ok,
        "num_fail": len(errors),
        "stable": representatives["stable"] is not None,
        "unique": unique_candidate is not None,
        "novel": representatives["novel"] is not None,
        "stable_novel": representatives["stable_novel"] is not None,
        "stable_unique_novel": False,
        "sun": False,
        "num_stable_novel_candidates": sum(
            bool(candidate["stable_novel"])
            for candidate in evaluated_candidates
        ),
        "num_unique_candidates": 0,
        "num_sun_candidates": 0,
        "num_sun_duplicates_removed": 0,
        "num_additional_unique_candidates_not_recounted": 0,
        "rep_stable": representatives["stable"],
        "rep_unique": unique_candidate,
        "rep_novel": representatives["novel"],
        "rep_stable_novel": representatives["stable_novel"],
        "rep_stable_unique_novel": representatives["stable_unique_novel"],
        "candidate_uniqueness_audit": [],
        "_evaluated_candidates": evaluated_candidates,
    }
    return {"record": record, "errors": errors}


def apply_sun_uniqueness(
    records: dict[int, dict[str, Any]],
    matcher_kind: str,
    *,
    matcher: StructureMatcherProtocol | None = None,
    structure_loader: Callable[[str | Path], Any] = load_evaluated_structure,
) -> None:
    stable_novel_pool: list[dict[str, Any]] = []
    for input_number in sorted(records):
        record = records[input_number]
        evaluated = list(record.pop("_evaluated_candidates", []))
        for candidate in evaluated:
            if not candidate["stable_novel"]:
                continue
            pooled = dict(candidate)
            pooled["input_number"] = input_number
            pooled["pretty_formula"] = record.get("pretty_formula", "")
            pooled["atoms"] = record.get("atoms", 0)
            stable_novel_pool.append(pooled)

    annotated, _ = (
        filter_unique_candidates(
            stable_novel_pool,
            matcher_kind,
            matcher=matcher,
            structure_loader=structure_loader,
        )
        if stable_novel_pool
        else ([], [])
    )
    annotated_by_input: dict[int, list[dict[str, Any]]] = {}
    for candidate in annotated:
        annotated_by_input.setdefault(int(candidate["input_number"]), []).append(
            candidate
        )

    for input_number in sorted(records):
        record = records[input_number]
        candidates = sorted(
            annotated_by_input.get(input_number, []),
            key=lambda item: (int(item["candidate_rank"]), str(item["filename"])),
        )
        sun_candidates = [candidate for candidate in candidates if candidate["unique"]]
        representative = dict(sun_candidates[0]) if sun_candidates else None
        record["stable_unique_novel"] = representative is not None
        record["sun"] = representative is not None
        record["num_unique_candidates"] = len(sun_candidates)
        record["num_sun_candidates"] = int(representative is not None)
        record["num_sun_duplicates_removed"] = len(candidates) - len(
            sun_candidates
        )
        record["num_additional_unique_candidates_not_recounted"] = max(
            0, len(sun_candidates) - 1
        )
        record["rep_stable_unique_novel"] = representative
        record["candidate_uniqueness_audit"] = [
            {
                "filename": candidate["filename"],
                "candidate_rank": int(candidate["candidate_rank"]),
                "match_structure_path": candidate["match_structure_path"],
                "match_structure_stage": candidate["match_structure_stage"],
                "stable": True,
                "novel": True,
                "stable_novel": True,
                "unique": bool(candidate["unique"]),
                "sun": bool(
                    representative is not None
                    and candidate["filename"] == representative["filename"]
                    and int(candidate["candidate_rank"])
                    == int(representative["candidate_rank"])
                ),
                "duplicate_of_input_number": candidate[
                    "duplicate_of_input_number"
                ],
                "duplicate_of_candidate_rank": candidate[
                    "duplicate_of_candidate_rank"
                ],
                "duplicate_of_filename": candidate["duplicate_of_filename"],
            }
            for candidate in candidates
        ]


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
        "stable_unique_novel",
        "sun",
        "num_stable_novel_candidates",
        "num_unique_candidates",
        "num_sun_candidates",
        "num_sun_duplicates_removed",
        "num_additional_unique_candidates_not_recounted",
        "rep_stable",
        "rep_unique",
        "rep_novel",
        "rep_stable_novel",
        "rep_stable_unique_novel",
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
            stable_unique_novel = record.get("rep_stable_unique_novel")
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
                    "stable_unique_novel": int(
                        bool(record.get("stable_unique_novel"))
                    ),
                    "sun": int(bool(record.get("sun"))),
                    "num_stable_novel_candidates": record.get(
                        "num_stable_novel_candidates", 0
                    ),
                    "num_unique_candidates": record.get(
                        "num_unique_candidates", 0
                    ),
                    "num_sun_candidates": record.get(
                        "num_sun_candidates", 0
                    ),
                    "num_sun_duplicates_removed": record.get(
                        "num_sun_duplicates_removed", 0
                    ),
                    "num_additional_unique_candidates_not_recounted": record.get(
                        "num_additional_unique_candidates_not_recounted", 0
                    ),
                    "rep_stable": stable.get("filename", "") if stable else "",
                    "rep_unique": unique.get("filename", "") if unique else "",
                    "rep_novel": novel.get("filename", "") if novel else "",
                    "rep_stable_novel": (
                        stable_novel.get("filename", "") if stable_novel else ""
                    ),
                    "rep_stable_unique_novel": (
                        stable_unique_novel.get("filename", "")
                        if stable_unique_novel
                        else ""
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
        for key in (
            "stable",
            "unique",
            "novel",
            "sun",
            "stable_novel",
            "stable_unique_novel",
        )
    }
    aggregate = {
        "candidate_budget": candidate_budget,
        "denominator": num_inputs,
        "numerators": numerators,
        "rates": {key: value / num_inputs for key, value in numerators.items()},
        "metric_definitions": {
            "stable": "any stable candidate with rank <= K",
            "unique": "any unique candidate with rank <= K",
            "novel": "any novel candidate with rank <= K",
            "stable_novel": "any single candidate with rank <= K that is both stable and novel before uniqueness filtering",
            "sun": (
                "the first candidate with rank <= K that is stable and novel "
                "and remains after StructureMatcher duplicate removal"
            ),
            "denominator": "complete official test split; an absent composition is unsuccessful",
        },
        "sun_uniqueness_rule": (
            "first select all stable-and-novel candidates, then sort them by "
            "candidate rank and input index and remove candidates matching an "
            "earlier retained candidate; the first remaining candidate per "
            "input supplies its single SUN hit"
        ),
        "candidate_failures": sum(record.get("num_fail", 0) for record in records.values()),
        "errors": errors,
    }
    with (out_dir / "aggregate_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best-of-K crystal candidates with MatterGen."
    )
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
    parser.add_argument("--relax", choices=("True",), default="True")
    parser.add_argument(
        "--structure-matcher", choices=("ordered", "disordered"), default="disordered"
    )
    parser.add_argument("--device")
    parser.add_argument("--potential-load-path")
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
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.relax != "True":
        raise ValueError(
            "--relax=False is unsupported because this wrapper does not accept "
            "per-candidate energies required for stability evaluation"
        )
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
    input_ranks: set[tuple[int, int]] = set()
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
        if rank < 1:
            invalid_names.append(str(cif_path))
            continue
        if rank > args.candidate_budget:
            continue
        input_rank = (input_number, rank)
        if input_rank in input_ranks:
            raise ValueError(
                f"Duplicate candidate rank {rank} for input {input_number}"
            )
        input_ranks.add(input_rank)
        item_metadata = (formula, num_atoms)
        if input_number in metadata and metadata[input_number] != item_metadata:
            raise ValueError(
                f"Conflicting formula or atom count for input {input_number}: "
                f"{metadata[input_number]!r} versus {item_metadata!r}"
            )
        groups.setdefault(input_number, []).append((rank, str(cif_path)))
        metadata.setdefault(input_number, item_metadata)
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
        "extra_args": args.extra,
    }
    records: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    futures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
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
                    "stable_unique_novel": False,
                    "sun": False,
                    "num_stable_novel_candidates": 0,
                    "num_unique_candidates": 0,
                    "num_sun_candidates": 0,
                    "num_sun_duplicates_removed": 0,
                    "num_additional_unique_candidates_not_recounted": 0,
                    "rep_stable": None,
                    "rep_unique": None,
                    "rep_novel": None,
                    "rep_stable_novel": None,
                    "rep_stable_unique_novel": None,
                    "candidate_uniqueness_audit": [],
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

    apply_sun_uniqueness(records, args.structure_matcher)

    if args.copy_cifs:
        for label, key in (
            ("stable", "rep_stable"),
            ("novel", "rep_novel"),
            ("stable_novel", "rep_stable_novel"),
            ("stable_unique_novel", "rep_stable_unique_novel"),
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
