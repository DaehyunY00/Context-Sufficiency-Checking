#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-cs_mps312}"

echo "[설치] conda 환경 생성: ${ENV_NAME} (Python 3.12)"
conda create -y -n "${ENV_NAME}" python=3.12

echo "[설치] pip 업그레이드"
conda run -n "${ENV_NAME}" python -m pip install -U pip

echo "[설치] PyTorch + 의존성 설치"
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

echo "[점검] MPS 장치 확인"
conda run -n "${ENV_NAME}" python experiments/check_device.py --config configs/llama3_8b_hotpot.yaml

echo ""
echo "[완료] 활성화 명령:"
echo "  conda activate ${ENV_NAME}"
