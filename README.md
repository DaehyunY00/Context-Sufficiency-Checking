# Context Sufficiency Checking 기반 로컬 RAG 실험 코드

MacBook Pro(M4) 로컬 환경(CPU/MPS) 우선으로 동작하도록 만든 연구용 프로토타입입니다.

## 1) 연구 목표
- 질문과 검색 문맥(top-k)의 충분성을 먼저 판정
- 충분하면 답변 생성, 불충분하면 재검색 또는 답변 거부("모르겠습니다.")
- Baseline 대비 환각률/커버리지/선택적 정확도 trade-off 측정

## 2) 디렉터리 구조
```text
Context_Sufficiency/
  retrieval/
    embedder.py
    index.py
    utils.py
  sufficiency/
    base.py
    llm_checker.py
    heuristic.py
    self_consistency.py
    entailment_checker.py
  generator/
    model.py
    utils.py
  evaluation/
    metrics.py
    evaluator.py
  experiments/
    run_baseline.py
    run_sufficiency.py
    ablations.py
  pipeline.py
  configs/
    default.yaml
  templates/
    autorater_ko.txt
  README.md
  requirements.txt
```

## 3) 설치 방법 (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Mac MPS 사용 권장 환경변수:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Mac GPU(MPS) 점검
먼저 아래 명령으로 현재 환경에서 MPS가 실제 사용 가능한지 확인하세요.
```bash
python experiments/check_device.py --config configs/default.yaml
```

설정:
- `run.device_preference: ["mps", "cpu"]`이면 MPS 우선, 실패 시 CPU 폴백
- `run.force_mps: true`이면 MPS가 불가능할 때 즉시 에러로 중단

## 4) 데이터 다운로드/준비
별도 수동 다운로드는 필요 없습니다.
- 첫 실행 시 `datasets`가 HotpotQA(`hotpot_qa/distractor`)를 자동 다운로드합니다.

## 5) 인덱스 생성
별도 스크립트 없이 실험 실행 시 자동 생성됩니다.
- 문단 청크 생성 → 임베딩 → FAISS CPU 인덱스 구축

## 6) Baseline 실행
```bash
python experiments/run_baseline.py \
  --config configs/default.yaml \
  --run-name baseline \
  --max-questions 300
```

## 7) Sufficiency 실행
```bash
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_heuristic_abstain \
  --checker heuristic \
  --strategy abstain \
  --max-questions 300
```

체커 옵션:
- `heuristic`
- `autorater`
- `self_consistency`
- `entailment` (옵션)

전략 옵션:
- `abstain`
- `reretrieve`
- `hybrid`

## 8) Ablation 실행
```bash
python experiments/ablations.py \
  --config configs/default.yaml \
  --run-name ablations \
  --max-questions 300 \
  --threshold-sweep 0.2,0.35,0.5,0.65 \
  --k-sweep 3,5,7
```

## 9) 결과 저장 경로
기본 출력 경로: `outputs`

저장 파일:
- 샘플 로그 JSONL: `outputs/<run_name>.jsonl`
- 요약 CSV: `outputs/<run_name>_summary.csv`
- 요약 Markdown: `outputs/<run_name>_summary.md`

## 10) JSONL 필수 필드
각 샘플은 아래 필드를 포함합니다.
- `question_id`, `question`, `gold_answer`
- `retrieved_doc_ids`, `retrieved_scores`
- `checker_name`, `checker_label`, `checker_score`
- `strategy_used`
- `final_answer`, `is_correct`, `is_abstain`
- `latency_ms`

## 11) 모델 설정
기본 설정(`configs/default.yaml`):
- Retriever 임베딩: `intfloat/e5-small-v2`
- Generator: `google/flan-t5-large`
- Autorater: `google/flan-t5-base`
- 장치 우선순위: `mps -> cpu`

Colab GPU 확장 시 `generator.model_name`을 아래로 교체 가능:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Meta-Llama-3-8B-Instruct`

## 12) Autorater 템플릿
`templates/autorater_ko.txt`를 그대로 사용하며, 코드에서 `{question}`, `{context}`만 치환합니다.
파싱 실패 시 `INSUFFICIENT`로 폴백합니다.
