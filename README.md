# Context Sufficiency Checking 기반 로컬 RAG 실험 코드

MacBook Pro(M4) 로컬 환경(CPU/MPS) 우선으로 동작하도록 만든 연구용 프로토타입입니다.

## 1) 연구 목표
- 질문과 검색 문맥(top-k)의 충분성을 먼저 판정
- 충분하면 답변 생성, 불충분하면 재검색 또는 답변 거부("모르겠습니다.")
- Baseline 대비 환각률/커버리지/선택적 정확도 trade-off 측정

### Answerable 운영정의(Operational Definition)
본 코드에서 CSC 점수는 `g(q, C_k) ≈ P(Answerable | q, C_k)`로 해석합니다.

- 수식 정의: `Answerable = 1 ⟺ ∃ a* supported by C_k`
- 실험 기본 구현: `evaluation.answerable.mode: gold_containment`
  - 정답 문자열(정규화)이 초기 검색 문맥 `C_k`에 포함되면 `Answerable=1`
- 대체 구현: `evaluation.answerable.mode: entailment`
  - NLI entailment 점수 `>= evaluation.answerable.entail_prob_threshold` 이면 `Answerable=1`

현재 사용 기준은 JSONL의 `oracle_answerable_mode`/`oracle_answerable_score`에 함께 기록됩니다.

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

또는 conda(Python 3.12) 권장:
```bash
conda create -y -n cs_py312 python=3.12
conda run -n cs_py312 python -m pip install -r requirements.txt
```

Mac MPS 사용 권장 환경변수:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

HuggingFace 캐시는 기본적으로 프로젝트 하위 `.hf_cache`를 사용합니다.
권한 이슈가 있으면 `configs/default.yaml`의 `run.hf_cache_dir`를 원하는 경로로 변경하세요.

### Mac GPU(MPS) 점검
먼저 아래 명령으로 현재 환경에서 MPS가 실제 사용 가능한지 확인하세요.
```bash
python experiments/check_device.py --config configs/default.yaml
```

설정:
- `run.device_preference: ["mps", "cpu"]`이면 MPS 우선, 실패 시 CPU 폴백
- `run.force_mps: true`이면 MPS가 불가능할 때 즉시 에러로 중단

MPS가 `False`로 나오면:
- `torch`가 nightly/dev 버전인지 확인 (`python -c "import torch; print(torch.__version__)"`)
- nightly 사용 중이면 안정 버전(`torch`, `torchvision`, `torchaudio` 동일 라인)으로 재설치 후 재점검 권장
- OS/파이썬 조합 이슈가 있으면 CPU 폴백으로 실험은 계속 가능

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
  --max-questions 500
```

## 7) Sufficiency 실행
```bash
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_heuristic_abstain \
  --checker heuristic \
  --strategy abstain \
  --max-questions 500
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
- `uncertainty_abstain`

불확실성 기반 abstention 예시:
```bash
# log-prob 기반 confidence (avg_token_prob)
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_uncertainty_logprob \
  --strategy uncertainty_abstain \
  --uncertainty-metric avg_token_prob \
  --uncertainty-threshold 0.20 \
  --max-questions 500

# entropy 기반 confidence
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_uncertainty_entropy \
  --strategy uncertainty_abstain \
  --uncertainty-metric entropy_confidence \
  --uncertainty-threshold 0.35 \
  --max-questions 500
```

## 8) Ablation 실행
```bash
python experiments/ablations.py \
  --config configs/default.yaml \
  --run-name ablations_v3 \
  --seeds 42,43,44 \
  --max-questions 500 \
  --threshold-sweep 0.2,0.35,0.5,0.65 \
  --k-sweep 3,5,7
```

autorater 사전점검(기본 활성화):
- ablation 시작 시 `--autorater-preflight-samples` 개수(기본 20개)로 JSON 파싱 성공률을 먼저 확인
- 성공률이 `--autorater-min-parse-success`(기본 0.30) 미만이면 autorater 실험은 자동 스킵
- 강제 실행이 필요하면 `--autorater-force-run` 추가
- autorater 파서는 `strict JSON 한 줄`만 허용하며, 코드블록/설명/조각 JSON은 모두 parse-fail 처리

## 9) CSC 확률 추정 분석(신규)
`ablations.py`는 아래 분석을 자동 저장합니다.
- `3-seed 반복`: 방법별 평균 ± 표준편차, CI
- `통계 검정`: baseline 대비 paired permutation test 기반 p-value
- `Calibration`: temperature scaling 전/후 reliability CSV/PNG, ECE/Brier(before/after), calibrated temperature
- `ROC/PR`: AUROC/AUPRC + 곡선 CSV/PNG
- `Risk-Coverage`: Risk(1-Accuracy)–Coverage 곡선 및 `AURC`
- `Sufficiency vs Retrieval`: 고검색-불충분비율, 저검색-충분비율, Pearson r, Spearman ρ, scatter plot
- `Mutual Information`: `I(Retrieval score; Sufficiency)` 정량화
- `CSC vs Final Accuracy`: CSC score와 최종 정답 여부 상관(Pearson/Spearman)
- `Uncertainty baseline`: log-prob/entropy 기반 abstain 비교표 (`F1`, `Hallucination`, `AURC`)
- `τ sweep 메인표`: `|τ|F1|Hallucination|AURC|` 형식의 threshold 비교
- `Self-consistency 안정성`: `n=3 vs n=5` 비교(분산 감소 수치 포함)
- `정책 최적화`: threshold `τ` sweep + 비용함수 기반 최적 `τ*` + bootstrap 95% CI
- `지연 분석`: warm-up 제거 평균, P50/P95, 표준편차, latency histogram, CPU/MPS 분리 평균 + 95% CI

## 10) 결과 저장 경로
기본 출력 경로: `outputs`

저장 파일:
- 샘플 로그 JSONL: `outputs/<run_name>.jsonl`
- 시드별 요약 CSV: `outputs/<run_name>_per_seed_summary.csv`
- 요약 CSV: `outputs/<run_name>_summary.csv`
- 요약 Markdown: `outputs/<run_name>_summary.md`
- Sufficiency vs Retrieval 분석: `outputs/<run_name>_retrieval_vs_sufficiency.md`
- Uncertainty 비교표: `outputs/<run_name>_uncertainty_comparison.md`
- Calibration 전/후 비교표: `outputs/<run_name>_calibration_before_after.md`
- AURC 메인 비교표: `outputs/<run_name>_aurc_main.md`
- τ sweep 메인표: `outputs/<run_name>_tau_sweep_main.md`
- Self-consistency 안정성: `outputs/<run_name>_self_consistency_stability.md`
- 지연 분석 요약표(CI 포함): `outputs/<run_name>_latency_device_ci.md`
- CSC-정답 상관 분석: `outputs/<run_name>_csc_accuracy_correlation.md`
- 정책 최적화 분석: `outputs/<run_name>_policy_optimization.md`
- 체커 곡선/캘리브레이션: `outputs/진단곡선/*`
- 지연 분석 산출물: `outputs/지연분석/*`

요약표에는 다음 진단 컬럼이 포함됩니다.
- `체커파싱성공률`
- `체커파싱실패수`
- `AURC`, `CSC_Temperature`, `CSC_ECE_before`, `CSC_ECE_after`, `CSC_Brier_before`, `CSC_Brier_after`
- `CSC_AUROC`, `CSC_AUPRC`
- `CSC-정답상관_Pearsonr`, `CSC-정답상관_Spearmanrho`
- `고검색-불충분비율`, `저검색-충분비율`
- `검색점수-충분성_Pearsonr`, `검색점수-충분성_Spearmanrho`
- `검색점수-충분성_MI`
- `평균지연(ms,warmup제외)`, `지연P50(ms)`, `지연P95(ms)`, `CPU평균지연(ms)`, `MPS평균지연(ms)`
- 집계 Markdown/CSV 모두 `EM/F1/환각률/커버리지`의 `평균±표준편차`와 `95% CI`를 함께 기록

## 11) JSONL 필수 필드
각 샘플은 아래 필드를 포함합니다.
- `question_id`, `question`, `gold_answer`
- `initial_retrieved_doc_ids`, `initial_retrieved_scores`
- `retrieved_doc_ids`, `retrieved_scores`
- `checker_name`, `checker_label`, `checker_score`
- `estimated_answerable_prob`, `oracle_answerable`
- `oracle_answerable_mode`, `oracle_answerable_score`, `oracle_answerable_matched_answer`
- `strategy_used`
- `final_answer`, `is_correct`, `is_abstain`
- `latency_ms`
- `uncertainty_metric`, `uncertainty_threshold`
- `generation_avg_token_logprob`, `generation_avg_token_prob`
- `generation_avg_token_entropy`, `generation_entropy_confidence`

## 12) 모델 설정
기본 설정(`configs/default.yaml`):
- Retriever 임베딩: `intfloat/e5-small-v2`
- Generator: `google/flan-t5-large`
- Autorater: `google/flan-t5-large` (오프라인 캐시 재사용 우선)
- Heuristic 기본 임계값: `min_coverage_ratio=0.5`
- Self-consistency 기본 샘플 수: `n_samples=5` (코드에서도 최소 5 보장)
- 장치 우선순위: `mps -> cpu`

Colab GPU 확장 시 `generator.model_name`을 아래로 교체 가능:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Meta-Llama-3-8B-Instruct`

## 13) 데이터셋 확장
기본: `hotpotqa`

추가 지원: `2wikimultihopqa`
추가 지원: `natural_questions`

빠른 실행용 예시 설정 파일: `configs/2wiki.yaml`
빠른 실행용 예시 설정 파일: `configs/nq.yaml`

예시:
```yaml
dataset:
  name: "2wikimultihopqa"
  hf_name: "scholarly-shadows-syndicate/2wikimultihopqa"
  hf_config: null
  split: "validation"
  max_questions: 500
```

Natural Questions 예시:
```yaml
dataset:
  name: "natural_questions"
  hf_name: "natural_questions"
  hf_config: "default"
  split: "validation"
  max_questions: 300
```

## 14) Autorater 템플릿
`templates/autorater_ko.txt`를 그대로 사용하며, 코드에서 `{question}`, `{context}`만 치환합니다.
파싱 실패 시 `INSUFFICIENT`로 폴백하며, 자동 재시도(`autorater.max_parse_retries`) 후에도 실패하면 JSONL의 `checker_meta`에 파싱오류를 기록합니다.
