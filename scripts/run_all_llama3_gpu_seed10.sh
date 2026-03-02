#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

trap 'rc=$?; log "[오류] line=${LINENO}, exit=${rc}, cmd=${BASH_COMMAND}"; exit ${rc}' ERR

# 사용법:
#   bash scripts/run_all_llama3_gpu_seed10.sh [transformers|ollama] [max_questions]
# 예:
#   bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
#   CLEAN_OLD=1 bash scripts/run_all_llama3_gpu_seed10.sh ollama 3000

BACKEND="${1:-transformers}"
MAX_Q="${2:-3000}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
RUN_PREFIX="${RUN_PREFIX:-llama3_seed10_3k}"
OUT_DIR="${OUT_DIR:-outputs}"
LOG_DIR="${OUT_DIR}/logs"
CLEAN_OLD="${CLEAN_OLD:-0}"
PY="${PYTHON_BIN:-python}"
SKIP_PIPELINE_CHECK="${SKIP_PIPELINE_CHECK:-1}"
MPS_CHECK_CONFIG="${MPS_CHECK_CONFIG:-configs/default.yaml}"
EXTRA_VALIDATIONS="${EXTRA_VALIDATIONS:-0}"  # 0: crosscheck/retriever_generalization skip
AUTORATER_PREFLIGHT_SAMPLES="${AUTORATER_PREFLIGHT_SAMPLES:-4}"
AUTORATER_MIN_PARSE_SUCCESS="${AUTORATER_MIN_PARSE_SUCCESS:-0.30}"
CHECKERS="${CHECKERS:-heuristic,autorater,self_consistency}"

if [[ "${BACKEND}" != "transformers" && "${BACKEND}" != "ollama" ]]; then
  echo "[오류] BACKEND는 transformers 또는 ollama 여야 합니다."
  exit 1
fi

if [[ "${BACKEND}" == "transformers" ]]; then
  CONFIGS="configs/llama3_8b_hotpot.yaml,configs/llama3_8b_2wiki.yaml,configs/llama3_8b_musique.yaml,configs/llama3_8b_strategyqa.yaml"
  HOTPOT_CFG="configs/llama3_8b_hotpot.yaml"
  TWOWIKI_CFG="configs/llama3_8b_2wiki.yaml"
  HOTPOT_TAG="hotpot"
else
  CONFIGS="configs/llama3_8b_hotpot_ollama.yaml,configs/llama3_8b_2wiki_ollama.yaml,configs/llama3_8b_musique_ollama.yaml,configs/llama3_8b_strategyqa_ollama.yaml"
  HOTPOT_CFG="configs/llama3_8b_hotpot_ollama.yaml"
  TWOWIKI_CFG="configs/llama3_8b_2wiki_ollama.yaml"
  HOTPOT_TAG="hotpot_ollama"
fi

mkdir -p "${LOG_DIR}"

log "[1/7] 장치 점검"
${PY} experiments/check_device.py --config "${HOTPOT_CFG}" 2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${BACKEND}_device.log"
if [[ "${SKIP_PIPELINE_CHECK}" == "1" ]]; then
  log "[1/7] 파이프라인 상세 점검 생략(SKIP_PIPELINE_CHECK=1)"
else
  log "[1/7] 파이프라인 상세 점검 실행(config=${MPS_CHECK_CONFIG})"
  ${PY} experiments/check_mps_pipeline.py --config "${MPS_CHECK_CONFIG}" --skip-entailment 2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${BACKEND}_pipeline_device.log"
fi

if [[ "${CLEAN_OLD}" == "1" ]]; then
  log "[2/7] 기존 불필요 outputs 정리"
  ${PY} scripts/clean_outputs_keep_prefix.py \
    --outputs-dir "${OUT_DIR}" \
    --keep-prefixes "${RUN_PREFIX},flare_tradeoff,gh1_bias_check,manual_paraphrase_subset" \
    --yes 2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${BACKEND}_cleanup.log"
else
  log "[2/7] 기존 결과 정리 스킵 (CLEAN_OLD=1로 활성화 가능)"
fi

# 정리 단계에서 logs 폴더가 제거될 수 있으므로 재생성
mkdir -p "${LOG_DIR}"

log "[3/7] Llama3-8B 10-seed 메인 실험 실행"
if [[ "${EXTRA_VALIDATIONS}" == "1" ]]; then
  ${PY} experiments/run_llama3_seed10_suite.py \
    --configs "${CONFIGS}" \
    --run-prefix "${RUN_PREFIX}" \
    --seeds "${SEEDS}" \
    --checkers "${CHECKERS}" \
    --max-questions "${MAX_Q}" \
    --autorater-preflight-samples "${AUTORATER_PREFLIGHT_SAMPLES}" \
    --autorater-min-parse-success "${AUTORATER_MIN_PARSE_SUCCESS}" \
    --include-bm25-threshold-baseline \
    --include-random-matched-baseline \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${BACKEND}_suite.log"
else
  ${PY} experiments/run_llama3_seed10_suite.py \
    --configs "${CONFIGS}" \
    --run-prefix "${RUN_PREFIX}" \
    --seeds "${SEEDS}" \
    --checkers "${CHECKERS}" \
    --max-questions "${MAX_Q}" \
    --autorater-preflight-samples "${AUTORATER_PREFLIGHT_SAMPLES}" \
    --autorater-min-parse-success "${AUTORATER_MIN_PARSE_SUCCESS}" \
    --include-bm25-threshold-baseline \
    --include-random-matched-baseline \
    --skip-answerable-crosscheck \
    --skip-retriever-generalization \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${BACKEND}_suite.log"
fi

log "[4/7] FLARE-lite trade-off (Hotpot)"
${PY} experiments/run_flare_tradeoff.py \
  --config "${HOTPOT_CFG}" \
  --run-name "flare_tradeoff_${RUN_PREFIX}_${HOTPOT_TAG}" \
  --seeds "${SEEDS}" \
  --max-questions "${MAX_Q}" \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/flare_tradeoff_${RUN_PREFIX}_${HOTPOT_TAG}.log"

log "[5/7] FLARE-lite trade-off (2Wiki)"
${PY} experiments/run_flare_tradeoff.py \
  --config "${TWOWIKI_CFG}" \
  --run-name "flare_tradeoff_${RUN_PREFIX}_2wiki" \
  --seeds "${SEEDS}" \
  --max-questions "${MAX_Q}" \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/flare_tradeoff_${RUN_PREFIX}_2wiki.log"

log "[6/7] gH1 수동검증 subset 생성"
JSONL_PATH="${OUT_DIR}/${RUN_PREFIX}_${HOTPOT_TAG}_seed42_heuristic_abstain.jsonl"
if [[ -f "${JSONL_PATH}" ]]; then
  ${PY} experiments/build_manual_paraphrase_subset.py \
    --jsonl "${JSONL_PATH}" \
    --output-csv "${OUT_DIR}/manual_paraphrase_subset_${RUN_PREFIX}.csv" \
    --sample-size 200 \
    --seed 42
  log "[안내] ${OUT_DIR}/manual_paraphrase_subset_${RUN_PREFIX}.csv 의 human_judgment(0/1) 수동 입력 후 7단계 실행"
else
  log "[경고] JSONL 파일이 없어 수동 subset 생성을 건너뜁니다: ${JSONL_PATH}"
fi

log "[7/7] gH1 편향 점검 실행 명령 안내"
cat <<EOF
${PY} experiments/run_gh1_bias_check.py \
  --jsonl ${OUT_DIR}/${RUN_PREFIX}_${HOTPOT_TAG}_seed42_heuristic_abstain.jsonl \
  --manual-csv ${OUT_DIR}/manual_paraphrase_subset_${RUN_PREFIX}.csv \
  --output-csv ${OUT_DIR}/gh1_bias_check_${RUN_PREFIX}.csv \
  --output-md ${OUT_DIR}/gh1_bias_check_${RUN_PREFIX}.md
EOF

log "[완료] 전체 파이프라인 실행 종료"
