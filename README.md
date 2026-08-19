# NextCrystal

NextCrystal is a symmetry-aware crystal-structure generation framework with two generation modes:

1. **Composition-only generation:** provide a chemical composition; NextCrystal predicts the most likely space group constraints and Wyckoff assignments, and explores structures under each predicted symmetry.
2. **Space-group-constrained generation:** provide a composition together with a target space group; NextCrystal can generate structures directly under the user-specified symmetry constraint.

## Two generation modes

### 1. From composition alone

For an input containing only `pretty_formula` and `NAtoms`, the Stage-A model
predicts five candidate space groups. Each composition–space-group pair is then
passed independently through Wyckoff assignment and structure generation. This
mode is useful when the crystal symmetry is unknown.

### 2. From composition with a target space group

When a target space group is known or needs to be explored, supply its
international `Spacegroup Number` together with the composition. NextCrystal
automatically resolves the canonical symbol from `data/wyckoff_template.json`.
The supplied group is a **generation constraint**, not a post-generation
filter:

- the Wyckoff model is conditioned directly on the requested space group and
  its available multiplicities;
- constrained assignment retains only element–Wyckoff allocations compatible
  with that space group;
- NextDiff receives the same international space-group number and the selected
  Wyckoff-orbit tokens when sampling coordinates and lattice parameters.

The same composition may be listed multiple times with different target space
groups to explore symmetry-dependent polymorphs. If no multiplicity-compatible
assignment exists for a requested `(composition, NAtoms, space group)` tuple,
the pipeline reports that no valid assignment was found rather than silently
changing the constraint.


## Repository layout

```text
configs/                         Hydra training and inference configuration
data/mp_20/                      MP-20 train/validation/test splits
data/wyckoff_template.json       Static 230-space-group Wyckoff reference
src/                             Stage-A/Stage-B models and post-processing
generate/convert_format_json.py  NextDiff input conversion
generate/nextdiff/               Minimal NextDiff sampling runtime
evaluate/                        External MatterGen CLI bridge 
examples/input_small.csv         Composition-only example
examples/input_spacegroup.csv    Space-group-constrained example
run_pipeline.sh                  End-to-end inference and sampling
```


## Environment

The environment uses Python 3.10, PyTorch 2.4, CUDA 12.4,
PyTorch Lightning 2.4, PyG 2.7, and `torch-scatter` 2.1.2. Create it with:

```bash
conda env create -f environment.yml
conda activate nextcrystal
```

GPU builds of PyTorch, PyG, and `torch-scatter` are platform-specific. If the
solver cannot select the correct CUDA build, install those three packages using
their official platform instructions and then run `pip install -e .`.


## Training

The MP-20 `train.csv`, `val.csv`, and `test.csv` splits are bundled under
`data/mp_20/`. Run:

```bash
python -m src.train task=sg experiment=sg trainer=sg \
  data/sg=mp_20

python -m src.train task=wyckoff experiment=wyckoff trainer=wyckoff \
  data/wyckoff=mp_20
```

The validation-best models save as `artifacts/mp_20/spacegroup.ckpt` and
`artifacts/mp_20/wyckoff.ckpt`, respectively.

All checkpoints can be downloaded from [Google Drive]( https://drive.google.com/drive/folders/1d8eE-I_y9pNy4OsWYf0tlmku0VodItDZ?usp=sharing)


## Quick start

Run the complete pipeline on the small example:

```bash
cd /path/to/NextCrystal
./run_pipeline.sh examples/input_small.csv results/smoke
```

Generated CIF files are written under `results/smoke/sample_structures/`.
NextDiff performs 1,000 reverse-diffusion steps; a CUDA GPU is strongly
recommended. CPU execution is supported but can be slow.

To run only the inexpensive stages before diffusion sampling:

```bash
./run_pipeline.sh examples/input_small.csv results/smoke --skip-sampling
```

### Space-group-constrained example

The file `examples/input_spacegroup.csv` fixes GaTe to space group 62 (`Pnma`):

```csv
cif_name,pretty_formula,NAtoms,Spacegroup Number
gate_pnma,GaTe,8,62
```

Run the symmetry-conditioned stages directly, without invoking the
space-group predictor:

```bash
cd /path/to/NextCrystal
export NEXTCRYSTAL_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/generate/nextdiff"

RUN_DIR="$PWD/results/sg62_pnma"
mkdir -p "$RUN_DIR"

python -m src.predict_wyckoff predict/wyckoff=mp_20 \
  "predict.wyckoff.input_csv=$PWD/examples/input_spacegroup.csv" \
  "predict.wyckoff.output_csv=$RUN_DIR/wyckoff_predictions.csv"

python -m src.run_postprocess postprocess=default \
  "postprocess.input_csv=$RUN_DIR/wyckoff_predictions.csv" \
  "postprocess.output_csv=$RUN_DIR/assignments.csv"

python generate/convert_format_json.py \
  "$RUN_DIR/assignments.csv" \
  "$RUN_DIR/nextdiff_input.json" \
  --wy_tokens generate/wy_tokens_complete.json

python generate/nextdiff/scripts/sample.py \
  --model_path generate/nextdiff/model/mp_csp \
  --save_path "$RUN_DIR/sample_structures" \
  --json_file "$RUN_DIR/nextdiff_input.json"
```

The constrained CSV may contain any number of rows. International space-group
numbers lie in `[1, 230]`; canonical symbols are filled automatically.

## Individual stages

All commands should be launched from the repository root.

```bash
export NEXTCRYSTAL_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/generate/nextdiff"

python -m src.predict_sg predict/sg=mp_20
python -m src.predict_wyckoff predict/wyckoff=mp_20
python -m src.run_postprocess postprocess=default

python generate/convert_format_json.py \
  outputs/mp_20/postprocessed_assignments_from_top5_sg.csv \
  outputs/mp_20/mp_test.json \
  --wy_tokens generate/wy_tokens_complete.json

python generate/nextdiff/scripts/sample.py \
  --model_path generate/nextdiff/model/mp_csp \
  --save_path results/sample_structures \
  --json_file outputs/mp_20/mp_test.json
```

## MatterGen evaluation

MatterGen is an optional, separately installed evaluation dependency. No
MatterGen source files are bundled in this repository. To prepare the numeric
NextDiff outputs and run the external evaluator:

```bash
python evaluate/prepare_evaluate_inputs.py \
  --assignment-csv results/run/postprocessed_assignments_from_top5_sg.csv \
  --query-json results/run/nextdiff_input.json \
  --cif-dir results/run/sample_structures \
  --output-dir results/run/mattergen_inputs

mattergen-evaluate \
  --structures_path=results/run/mattergen_inputs \
  --relax=True \
  --structure_matcher=disordered \
  --reference_dataset_path=evaluate/reference/reference_MP2020correction.gz \
  --energy_correction_scheme=MP2020 \
  --save_as=results/run/mattergen_metrics.json
```

The MP2020 reference path is committed as a Git LFS pointer rather than an
873 MB repository blob. Install Git LFS and materialize the object before
evaluation.


## Acknowledgements

We sincerely thank the authors of
[DiffCSP++](https://github.com/jiaor17/DiffCSP-PP) for their excellent work,
which provided an important foundation for the symmetry-conditioned sampling
runtime.
