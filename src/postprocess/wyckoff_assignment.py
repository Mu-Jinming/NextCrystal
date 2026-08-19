# path: src/postprocess/wyckoff_assignment.py
from __future__ import annotations

import ast
import itertools
import json
import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm


MISSING_LETTER_EPS = 1e-12


@dataclass
class PostProcessConfig:
    input_csv: str
    output_csv: str
    wyckoff_template_path: str
    beam_width: int = 200
    candidate_count: int = 5
    max_workers: int = 24
    max_atom_compos: int = 600
    topn_column: str = "Top-6 Predicted Wyckoff Letters"
    sg_number_column: str = "Spacegroup Number"
    output_column: str = "Assignments"


def infer_dof_from_coordinates(coord_list):
    if not coord_list:
        return 0
    vars_found = set()
    for s in coord_list:
        if not isinstance(s, str):
            continue
        ss = s.lower()
        if re.search(r"(?<![a-z])x(?![a-z])", ss):
            vars_found.add("x")
        if re.search(r"(?<![a-z])y(?![a-z])", ss):
            vars_found.add("y")
        if re.search(r"(?<![a-z])z(?![a-z])", ss):
            vars_found.add("z")
    return len(vars_found)


def load_template_info(template_path: str):
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wyckoff_dict = {}
    for group in data["space_groups"]:
        number = int(group["number"])
        letter_info = {}
        for pos in group["wyckoff_positions"]:
            letter = pos["letter"]
            mult = int(pos["multiplicity"])
            coords = pos.get("coordinates", [])
            dof = infer_dof_from_coordinates(coords)
            letter_info[letter] = {
                "multiplicity": mult,
                "dof": dof,
                "coordinates": coords,
            }
        wyckoff_dict[number] = letter_info
    return wyckoff_dict


def get_wyckoff_positions(spacegroup_number, wyckoff_template):
    info = wyckoff_template.get(spacegroup_number, {})
    return {l: d["multiplicity"] for l, d in info.items()}


def get_wyckoff_dofs(spacegroup_number, wyckoff_template):
    info = wyckoff_template.get(spacegroup_number, {})
    return {l: d.get("dof", 0) for l, d in info.items()}


def create_atoms(topn_dict_list):
    atoms = defaultdict(list)
    for atom_id, atom_dict in enumerate(topn_dict_list):
        for element, letter_probs in atom_dict.items():
            atoms[element].append({"id": atom_id, "letters": letter_probs})
    return atoms


def canonical_assignment_key(assign_str: str):
    tokens = []
    if not isinstance(assign_str, str):
        return ()
    for block in assign_str.split(";"):
        for tok in block.split(","):
            tok = tok.strip()
            if tok and ":" in tok:
                tokens.append(tok)
    return tuple(sorted(tokens))


def expand_atom_letters_with_all_letters(atoms_list, all_letters, eps=MISSING_LETTER_EPS):
    for atom in atoms_list:
        lp = atom.get("letters", {})
        for l in all_letters:
            if l not in lp:
                lp[l] = eps
        atom["letters"] = lp


def beam_search_assignments_single_element(atoms, letter_multiplicity_dict, fixed_letters, beam_width=200, max_atom_compos=600):
    states = [([], frozenset(), 0.0, frozenset())]

    while True:
        new_states = []
        progress = False

        for assignment, assigned_atoms, log_p_sum, used_fixed in states:
            remaining_atoms = [atom for atom in atoms if atom["id"] not in assigned_atoms]
            if not remaining_atoms:
                new_states.append((assignment, assigned_atoms, log_p_sum, used_fixed))
                continue

            for letter, multiplicity in letter_multiplicity_dict.items():
                if letter in fixed_letters and letter in used_fixed:
                    continue

                eligible_atoms = [atom for atom in remaining_atoms if letter in atom["letters"]]
                if len(eligible_atoms) < multiplicity:
                    continue

                for combination in itertools.islice(itertools.combinations(eligible_atoms, multiplicity), max_atom_compos):
                    selected_ids = [atom["id"] for atom in combination]
                    log_prob = sum(math.log(atom["letters"][letter] + 1e-12) for atom in combination)

                    new_assignment = assignment + [(letter, selected_ids)]
                    new_assigned_atoms = assigned_atoms.union(selected_ids)
                    new_used_fixed = used_fixed.union([letter]) if letter in fixed_letters else used_fixed
                    new_states.append((new_assignment, new_assigned_atoms, log_p_sum + log_prob, new_used_fixed))
                    progress = True

        if not progress:
            break

        states = sorted(new_states, key=lambda x: -x[2])[:beam_width]

    completed_assignments = [
        (assignment, log_p_sum, used_fixed)
        for assignment, assigned_atoms, log_p_sum, used_fixed in states
        if len(assigned_atoms) == len(atoms)
    ]
    return completed_assignments


def process_row(row_dict, wyckoff_template, cfg: PostProcessConfig):
    try:
        space_group_number = int(row_dict[cfg.sg_number_column])
        letter_multiplicity_dict = get_wyckoff_positions(space_group_number, wyckoff_template)
        if not letter_multiplicity_dict:
            return [f"No Wyckoff template information for space group {space_group_number}."]

        letter_dof_dict = get_wyckoff_dofs(space_group_number, wyckoff_template)
        fixed_letters = {l for l, dof in letter_dof_dict.items() if dof == 0}
        all_letters = list(letter_multiplicity_dict.keys())

        topn_predicted = ast.literal_eval(row_dict[cfg.topn_column])
        atoms_dict = create_atoms(topn_predicted)

        final_assignments = {}

        for element, atoms in atoms_dict.items():
            assignments = beam_search_assignments_single_element(
                atoms=atoms,
                letter_multiplicity_dict=letter_multiplicity_dict,
                fixed_letters=fixed_letters,
                beam_width=cfg.beam_width,
                max_atom_compos=cfg.max_atom_compos,
            )

            if not assignments:
                atoms_relaxed = [{"id": a["id"], "letters": dict(a["letters"])} for a in atoms]
                expand_atom_letters_with_all_letters(atoms_relaxed, all_letters)
                assignments = beam_search_assignments_single_element(
                    atoms=atoms_relaxed,
                    letter_multiplicity_dict=letter_multiplicity_dict,
                    fixed_letters=fixed_letters,
                    beam_width=cfg.beam_width,
                    max_atom_compos=cfg.max_atom_compos,
                )

            if not assignments:
                return [f"Unable to find an assignment satisfying the Wyckoff multiplicity constraints (element {element})."]

            sorted_assignments = sorted(assignments, key=lambda x: -x[1])[: cfg.candidate_count]
            formatted_assignments = []
            for assignment, log_p_sum, used_fixed in sorted_assignments:
                atom_assignment = {}
                for letter, atom_ids in assignment:
                    for aid in atom_ids:
                        atom_assignment[aid] = {"element": element, "letter": letter}
                sorted_atom_ids = sorted(atom_assignment.keys())
                formatted_assignment = ", ".join(
                    f"{atom_assignment[aid]['element']}:{atom_assignment[aid]['letter']}"
                    for aid in sorted_atom_ids
                )
                formatted_assignments.append((formatted_assignment, log_p_sum, set(used_fixed)))

            final_assignments[element] = formatted_assignments

        combined_assignments = [("", 0.0, set())]
        for _element, assignments in final_assignments.items():
            new_combined_assignments = []
            for combined_assignment, combined_log_p_sum, used_fixed_global in combined_assignments:
                for assignment_str, log_p_sum, used_fixed_local in assignments:
                    if used_fixed_global.intersection(used_fixed_local):
                        continue
                    new_assign = f"{combined_assignment}; {assignment_str}" if combined_assignment else assignment_str
                    new_combined_assignments.append(
                        (new_assign, combined_log_p_sum + log_p_sum, used_fixed_global.union(used_fixed_local))
                    )
            combined_assignments = sorted(new_combined_assignments, key=lambda x: -x[1])[: cfg.beam_width]

        if not combined_assignments:
            return ["Unable to find a combined assignment satisfying the Wyckoff multiplicity constraints."]

        combined_assignments_sorted = sorted(combined_assignments, key=lambda x: -x[1])

        unique_assignments = []
        seen_keys = set()
        for assign_str, log_p_sum, _used_fixed in combined_assignments_sorted:
            key = canonical_assignment_key(assign_str)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_assignments.append((assign_str, log_p_sum))
            if len(unique_assignments) >= cfg.candidate_count:
                break

        final_assignments_str = [a for a, _ in unique_assignments]
        if not final_assignments_str:
            final_assignments_str = [assign for assign, _, _ in combined_assignments_sorted[: cfg.candidate_count]]

        return final_assignments_str

    except Exception as e:
        logging.exception("post-process row failed")
        return [f"Failed: {e}"]


class WyckoffAssignmentPostProcessor:
    def __init__(self, cfg: PostProcessConfig):
        self.cfg = cfg
        self.template = load_template_info(cfg.wyckoff_template_path)

    def run(self):
        df = pd.read_csv(self.cfg.input_csv)
        required_columns = [self.cfg.sg_number_column, "Predicted Wyckoff Letters", self.cfg.topn_column]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain columns: {required_columns}")

        results = [None] * len(df)

        if self.cfg.max_workers <= 1:
            for idx, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Postprocess",
                unit="row",
            ):
                results[idx] = process_row(row.to_dict(), self.template, self.cfg)
        else:
            with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
                futures = {
                    executor.submit(process_row, row.to_dict(), self.template, self.cfg): idx
                    for idx, row in df.iterrows()
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Postprocess",
                    unit="row",
                ):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = [f"Failed: {e}"]

        result_df = df.copy()
        result_df[self.cfg.output_column] = [
            json.dumps(x if isinstance(x, list) else ["Unknown Error"], ensure_ascii=False)
            for x in results
        ]
        result_df.to_csv(self.cfg.output_csv, index=False, encoding="utf-8")
        return result_df
