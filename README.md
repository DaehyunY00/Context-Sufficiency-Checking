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

Mac GPU 우선 PyTorch 재설치(권장):
```bash
python -m pip install -U "torch>=2.6.0" "torchvision>=0.21.0" "torchaudio>=2.6.0"
```

Python 3.13 환경에서 MPS 비활성 시, Python 3.12 환경 자동 구성:
```bash
bash scripts/setup_mps_env.sh cs_mps312
conda activate cs_mps312
```

HuggingFace 캐시는 기본적으로 프로젝트 하위 `.hf_cache`를 사용합니다.
권한 이슈가 있으면 `configs/default.yaml`의 `run.hf_cache_dir`를 원하는 경로로 변경하세요.

### Mac GPU(MPS) 점검
먼저 아래 명령으로 현재 환경에서 MPS가 실제 사용 가능한지 확인하세요.
```bash
python experiments/check_device.py --config configs/default.yaml
```

파이프라인 구성요소(임베딩/생성/NLI)까지 포함한 점검:
```bash
python experiments/check_mps_pipeline.py --config configs/default.yaml
```

설정:
- `run.device_preference: ["mps", "cpu"]`이면 MPS 우선, 실패 시 CPU 폴백
- `run.force_mps: true`이면 MPS가 불가능할 때 즉시 에러로 중단

MPS가 `False`로 나오면:
- `torch`가 nightly/dev 버전인지 확인 (`python -c "import torch; print(torch.__version__)"`)
- nightly 사용 중이면 안정 버전(`torch`, `torchvision`, `torchaudio` 동일 라인)으로 재설치 후 재점검 권장
- OS/파이썬 조합 이슈가 있으면 CPU 폴백으로 실험은 계속 가능

PyTorch/MPS가 계속 실패하면(대안):
1. Ollama(로컬 Metal backend) 설치/실행
```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M
```
2. `configs/llama3_8b_*_ollama.yaml` 사용(생성/autorater를 Ollama로 수행, retrieval/평가는 기존 파이프라인 유지)

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
- `retrieval_score_threshold_abstain` (`bm25_score_threshold` 별칭)
- `random_abstain`
- `flare_lite` (FLARE 경량 근사)

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

# retrieval score threshold 기반 abstain (BM25_score_threshold baseline)
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_bm25_threshold \
  --strategy retrieval_score_threshold_abstain \
  --bm25-threshold 0.50 \
  --max-questions 500

# random abstain baseline
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_random_abstain \
  --strategy random_abstain \
  --random-abstain-rate 0.10 \
  --max-questions 500

# FLARE-lite baseline
python experiments/run_sufficiency.py \
  --config configs/default.yaml \
  --run-name suff_flare_lite \
  --strategy flare_lite \
  --flare-metric avg_token_prob \
  --flare-confidence-threshold 0.35 \
  --flare-max-rounds 2 \
  --flare-k-growth 3 \
  --max-questions 500
```

## 8) Ablation 실행
```bash
python experiments/ablations.py \
  --config configs/default.yaml \
  --run-name ablations_v3 \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 500 \
  --threshold-sweep 0.2,0.35,0.5,0.65 \
  --k-sweep 3,5,7 \
  --include-bm25-threshold-baseline \
  --include-random-matched-baseline
```
참고: `BM25_score_threshold`, `Random_matched_abstain` baseline은 기본값으로 비활성화되어 있으며 위 `--include-*` 플래그를 줄 때만 실행됩니다.


autorater 사전점검(기본 활성화):
- ablation 시작 시 `--autorater-preflight-samples` 개수(기본 20개)로 JSON 파싱 성공률을 먼저 확인
- 성공률이 `--autorater-min-parse-success`(기본 0.30) 미만이면 autorater 실험은 자동 스킵
- 강제 실행이 필요하면 `--autorater-force-run` 추가
- autorater 파서는 `strict JSON 한 줄`만 허용하며, 코드블록/설명/조각 JSON은 모두 parse-fail 처리

## 9) CSC 확률 추정 분석(신규)
`ablations.py`는 아래 분석을 자동 저장합니다.
- `다중 시드 반복`: 방법별 평균 ± 표준편차, CI
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

### 9-1) 보조 실험 10-seed 확장 권장 실행
```bash
python experiments/ablations.py \
  --config configs/default.yaml \
  --run-name ablations_seed10_hotpot \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 120

python experiments/ablations.py \
  --config configs/2wiki.yaml \
  --run-name ablations_seed10_2wiki \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 120
```

### 9-1-b) FLARE-lite vs CSC trade-off (3,000+ 샘플)
```bash
python experiments/run_flare_tradeoff.py \
  --config configs/default.yaml \
  --run-name flare_tradeoff_hotpot_3k_seed10 \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 3000
```
- 전체 dev set 실행: `--max-questions 0` (split 전체 사용)
- 결과: `outputs/<run_name>_summary.csv`, `outputs/<run_name>_tradeoff.md`

### 9-2) Gold Containment vs NLI 교차검증(부록용)
```bash
python experiments/run_answerable_crosscheck.py \
  --config configs/default.yaml \
  --run-name answerable_cross_hotpot \
  --checker heuristic \
  --strategy abstain \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 120 \
  --entail-model cross-encoder/nli-distilroberta-base \
  --entail-threshold 0.6
```

### 9-3) 리트리버 일반화 검증(e5 vs DPR vs ColBERT-근사)
```bash
python experiments/run_retriever_generalization.py \
  --config configs/default.yaml \
  --run-name retriever_generalization_hotpot \
  --checker heuristic \
  --strategy abstain \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 120
```

참고:
- `DPR`은 dual-encoder(`facebook/dpr-question_encoder-single-nq-base`, `facebook/dpr-ctx_encoder-single-nq-base`)를 직접 지원합니다.
- `ColBERT`는 현재 late-interaction 완전 구현이 아닌 임베딩 근사 경로(`sentence_transformer`)로 동작합니다.

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
- Hallucination-Coverage 비교표: `outputs/<run_name>_hallucination_coverage_tradeoff.md`
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
- `initial_contexts`
- `retrieved_doc_ids`, `retrieved_scores`
- `retrieved_contexts`
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
추가 지원: `musique`
추가 지원: `strategyqa`

빠른 실행용 예시 설정 파일: `configs/2wiki.yaml`
빠른 실행용 예시 설정 파일: `configs/nq.yaml`
빠른 실행용 예시 설정 파일: `configs/musique.yaml`
빠른 실행용 예시 설정 파일: `configs/strategyqa.yaml`

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
  drop_empty_gold_answers: true  # 권장: 정답 공백 샘플 제외
```

MuSiQue 예시:
```yaml
dataset:
  name: "musique"
  hf_name: "dgslibisey/MuSiQue"
  split: "validation"
  max_questions: 300
```

StrategyQA 예시:
```yaml
dataset:
  name: "strategyqa"
  hf_name: "ChilleD/StrategyQA"
  split: "train"  # 미러별 split 차이 가능
  max_questions: 300
```

## 14) Llama3-8B 10-seed 재실험
Llama3-8B 설정 파일:
- `configs/llama3_8b_hotpot.yaml`
- `configs/llama3_8b_2wiki.yaml`
- `configs/llama3_8b_musique.yaml`
- `configs/llama3_8b_strategyqa.yaml`
- (대안) `configs/llama3_8b_hotpot_ollama.yaml`
- (대안) `configs/llama3_8b_2wiki_ollama.yaml`
- (대안) `configs/llama3_8b_musique_ollama.yaml`
- (대안) `configs/llama3_8b_strategyqa_ollama.yaml`

통합 실행(Hotpot/2Wiki/MuSiQue/StrategyQA + 10-seed):
```bash
python experiments/run_llama3_seed10_suite.py \
  --run-prefix llama3_seed10 \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 3000 \
  --include-bm25-threshold-baseline \
  --include-random-matched-baseline \
  --skip-answerable-crosscheck \
  --skip-retriever-generalization \
  --output-dir outputs
```

참고:
- `meta-llama/Meta-Llama-3.1-8B-Instruct`는 접근 권한이 필요할 수 있습니다(`hf auth login`).
- MPS 메모리가 부족하면 `max_questions`를 줄이거나 checkers를 `heuristic,self_consistency`로 제한하세요.

Ollama 대안 실행(예시):
```bash
python experiments/run_llama3_seed10_suite.py \
  --configs configs/llama3_8b_hotpot_ollama.yaml,configs/llama3_8b_2wiki_ollama.yaml,configs/llama3_8b_musique_ollama.yaml,configs/llama3_8b_strategyqa_ollama.yaml \
  --run-prefix llama3_seed10_3k_ollama \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --max-questions 3000 \
  --include-bm25-threshold-baseline \
  --include-random-matched-baseline \
  --skip-answerable-crosscheck \
  --skip-retriever-generalization \
  --output-dir outputs
```

## 15) gH1 편향 점검(gold_containment 연계성)
1) 수동 검증 subset 생성(약 200개):
```bash
python experiments/build_manual_paraphrase_subset.py \
  --jsonl outputs/llama3_seed10_hotpot_seed42_heuristic_abstain.jsonl \
  --output-csv outputs/manual_paraphrase_subset.csv \
  --sample-size 200
```
- 생성된 CSV의 `human_judgment` 컬럼을 사람이 `0/1`로 채웁니다.

2) gH1 vs gH1' AUROC 비교:
```bash
python experiments/run_gh1_bias_check.py \
  --jsonl outputs/llama3_seed10_hotpot_seed42_heuristic_abstain.jsonl \
  --manual-csv outputs/manual_paraphrase_subset.csv \
  --output-csv outputs/gh1_bias_check.csv \
  --output-md outputs/gh1_bias_check.md
```
- `gH1' = overlap(q \\ a*, C_k)`로 계산되며, `gold_containment`와 `human_judgment` 라벨 각각에서 AUROC를 비교합니다.

## 16) Autorater 템플릿
`templates/autorater_ko.txt`를 그대로 사용하며, 코드에서 `{question}`, `{context}`만 치환합니다.
파싱 실패 시 `INSUFFICIENT`로 폴백하며, 자동 재시도(`autorater.max_parse_retries`) 후에도 실패하면 JSONL의 `checker_meta`에 파싱오류를 기록합니다.

## 17) 논문용 전체 실행 (Llama3-8B, 10-seed, GPU 우선)
한 번에 실행(장치점검 → 기존결과정리 옵션 → 메인실험 → FLARE trade-off → gH1 수동 subset 생성):
```bash
bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
```

`[1/7]` 이후 멈춘 것처럼 보이면(대형 모델 점검 단계), 상세 점검을 생략하고 바로 본실험:
```bash
SKIP_PIPELINE_CHECK=1 bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
```

autorater 사전점검이 오래 걸릴 때(기본 4샘플, 더 줄이려면 1~2):
```bash
AUTORATER_PREFLIGHT_SAMPLES=2 SKIP_PIPELINE_CHECK=1 bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
```

PyTorch MPS가 계속 실패할 때(대안: Ollama/Metal):
```bash
bash scripts/run_all_llama3_gpu_seed10.sh ollama 3000
```

기존 불필요 outputs 먼저 정리:
```bash
CLEAN_OLD=1 bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
```

교차검증/리트리버 일반화까지 포함(추가 시간 소요):
```bash
EXTRA_VALIDATIONS=1 bash scripts/run_all_llama3_gpu_seed10.sh transformers 3000
```

`outputs` 정리만 단독 실행:
```bash
python scripts/clean_outputs_keep_prefix.py \
  --outputs-dir outputs \
  --keep-prefixes llama3_seed10_3k,flare_tradeoff,gh1_bias_check,manual_paraphrase_subset \
  --yes
```
