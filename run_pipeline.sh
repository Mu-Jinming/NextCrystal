#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_CSV="${1:-${ROOT}/examples/input_small.csv}"
RUN_DIR="${2:-${ROOT}/results/run}"
MODE="${3:-}"

export NEXTCRYSTAL_ROOT="${ROOT}"
export NEXTDIFF_ROOT="${ROOT}/generate/nextdiff"
export USE_WANDB_LOGGING=0
export PYTHONPATH="${ROOT}:${NEXTDIFF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SG_CSV="${RUN_DIR}/sg_topk_expanded.csv"
WYCKOFF_CSV="${RUN_DIR}/wyckoff_predictions_from_top5_sg.csv"
ASSIGNMENT_CSV="${RUN_DIR}/postprocessed_assignments_from_top5_sg.csv"
NEXTDIFF_JSON="${RUN_DIR}/nextdiff_input.json"
CIF_DIR="${RUN_DIR}/sample_structures"

mkdir -p "${RUN_DIR}"
cd "${ROOT}"

python -m src.predict_sg \
  predict/sg=mp_20 \
  "predict.sg.input_csv=${INPUT_CSV}" \
  "predict.sg.output_csv=${SG_CSV}"

python -m src.predict_wyckoff \
  predict/wyckoff=mp_20 \
  "predict.wyckoff.input_csv=${SG_CSV}" \
  "predict.wyckoff.output_csv=${WYCKOFF_CSV}"

python -m src.run_postprocess \
  postprocess=default \
  "postprocess.input_csv=${WYCKOFF_CSV}" \
  "postprocess.output_csv=${ASSIGNMENT_CSV}"

python generate/convert_format_json.py \
  "${ASSIGNMENT_CSV}" \
  "${NEXTDIFF_JSON}" \
  --wy_tokens "${ROOT}/generate/wy_tokens_complete.json"

if [[ "${MODE}" == "--skip-sampling" ]]; then
  exit 0
fi

mkdir -p "${CIF_DIR}"
python "${NEXTDIFF_ROOT}/scripts/sample.py" \
  --model_path "${NEXTDIFF_ROOT}/model/mp_csp" \
  --save_path "${CIF_DIR}" \
  --json_file "${NEXTDIFF_JSON}"
