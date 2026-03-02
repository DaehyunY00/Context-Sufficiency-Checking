

## 1. 제목(안)
- 로컬 컴퓨트 제약 하 문맥 충분성 기반 선택적 RAG 제어: 파일럿 실험
- A Pilot Study of Context Sufficiency Estimation for Selective RAG Control under Local Compute Constraints

## 2. 핵심 기여 문장 (논문 본문/커버레터 공용)
1. 본 연구는 문맥 충분성 검사를 이진 분류가 아니라 `g(q, C_k) ≈ P(Answerable | q, C_k)` 추정 문제로 재정의하였다.
2. heuristic, autorater, self-consistency를 각각 feature-based, semantic LLM-based, variance-based epistemic estimator로 통합 이론 틀에 정렬하였다.
3. RAG 성능을 relevance(검색 품질)와 answerability(문맥 충분성)로 분리 평가하고, 두 축이 일치하지 않는 사례를 정량적으로 제시하였다.
4. 메인 비교는 10-seed(`42~51`)로 확장하고, calibration(before/after), Risk-Coverage/AURC, k-sweep 보조 분석까지 포함한 로컬 Mac 기반 파일럿 실험 체계를 구현하여 전 로그를 공개하였다.

## 3. 섹션 구성(권장, 8~12쪽)
| 섹션 | 권장 비중 | 길이 가이드 |
|---|---:|---:|
| 1. 서론 | 15% | 1.2~1.8쪽 |
| 2. 관련 연구 | 12% | 1.0~1.4쪽 |
| 3. CSC의 이론적 정의 | 15% | 1.2~1.8쪽 |
| 4. CSC 추정기 설계 | 18% | 1.4~2.0쪽 |
| 5. 선택적 RAG 정책 모델 | 10% | 0.8~1.2쪽 |
| 6. 실험 설계 | 12% | 1.0~1.4쪽 |
| 7. 결과 및 고찰 | 13% | 1.0~1.6쪽 |
| 8. 결론 | 5% | 0.5~0.8쪽 |

## 4. 본문 템플릿

### 4.1 서론
RAG는 검색 문서를 활용하여 정답률을 높일 수 있으나, 검색된 문맥이 질문에 답하기에 충분하지 않을 경우 환각이 발생한다. 기존 연구는 검색 relevance 향상 또는 생성 품질 개선에 집중해 왔으며, 문맥의 answerability 자체를 제어 신호로 사용하는 연구는 제한적이다. 본 연구는 CSC(Context Sufficiency Checking)를 확률 추정 문제로 재정의하고, 추정 결과를 기반으로 생성·재검색·거부를 선택하는 정책을 제안하였다. 또한 로컬 환경(Mac CPU/MPS)에서 재현 가능한 실험 파이프라인을 구현하여 신뢰도 중심 RAG 설계의 실용성을 검증하였다.

### 4.2 관련 연구
관련 연구는 크게 (i) retrieval 개선, (ii) uncertainty-aware generation, (iii) selective prediction으로 구분된다. 특히 Joren et al.(ICLR 2025, *Sufficient Context*)은 context sufficiency를 전면에 제시하고, Gemini 1.5 Pro/GPT-4o/Claude 3.5 등 대형 API 모델 기반 autorater를 중심으로 충분성 신호의 유효성을 입증하였다. 이 계열 연구는 고성능 semantic autorater의 효과를 보여주었지만, 상용 API 의존성과 계산비용 측면에서 로컬 재현성 제약이 남는다.

본 연구의 포지셔닝은 해당 흐름과 경쟁이 아니라 보완이다. 즉, 대형 API autorater 중심의 선행연구를 로컬 컴퓨트 제약 환경(Mac CPU/MPS, 오픈소스 소형 모델, 무상용 API)으로 확장하여, heuristic/self-consistency estimator의 실용성과 한계를 체계적으로 분석한다. 또한 동일 파이프라인에서 calibration/ROC/AURC/latency를 함께 보고하여, “충분성 신호를 실제 운영 제약 하에서 어떻게 사용할 것인가”를 공학적으로 명확히 한다.  
추가로 selective prediction 관련 conformal prediction 계열(Angelopoulos & Bates, 2021; Traub et al., 2024)과 RAG 평가 프레임워크(RAGAS, FActScore)와의 연결을 명시해, 본 연구를 “정답률 개선”보다 “신뢰도 제어” 축에 배치한다.

### 4.3 CSC의 이론적 정의
본 연구는 CSC를 다음과 같이 정의한다.
\[
g(q, C_k) \approx P(\text{Answerable} \mid q, C_k)
\]
여기서 `Answerable`은 주어진 문맥만으로 검증 가능한 답변을 생성할 수 있는 사건이다. 최종 의사결정은 임계값 기반으로 수행한다.
\[
\hat{y}=
\begin{cases}
\text{SUFFICIENT}, & g(q, C_k) \ge \tau \\
\text{INSUFFICIENT}, & \text{otherwise}
\end{cases}
\]
`Answerable`의 운영정의는 다음과 같다.
\[
\text{Answerable}=1 \iff \exists a^* \text{ supported by } C_k
\]
실험 구현에서는 기본적으로 `gold answer` 문자열 포함 여부(`gold_containment`)를 사용하였다. 즉, 정규화된 정답 문자열이 검색 문맥 `C_k`에 포함되면 `Answerable=1`로 판정한다. 대체 구현으로는 NLI entailment 기반 판정(`entailment score \ge \tau_e`)을 지원하며, 본문에서는 실제 사용한 기준(`gold_containment` 또는 `entailment`)을 명시한다. 이 정의는 CSC를 완전한 확률 모델로 단정하기보다, answerability에 대한 근사적 확률 점수를 제공하는 uncertainty-aware scoring mechanism으로 해석할 근거를 제공한다.

### 4.4 CSC 추정기 설계
1. Heuristic estimator: 질문 키워드 커버리지를 feature로 사용하여 `P(Answerable|q,C_k)`를 근사한다.  
   본문 기본형(`g_H1`) 외에 TF-IDF 가중 커버리지(`g_H2`), 개체 매칭 비율(`g_H3`)을 부록에서 비교하였다(표 A.7).  
2. LLM semantic estimator(autorater): 질문-문맥 의미 적합성을 기반으로 JSON 고정 출력(`label`, `confidence`)을 생성한다.  
3. Self-consistency estimator: 동일 문맥에서 다중 샘플 생성의 분산을 epistemic uncertainty로 보고, 다수결 비율을 확률 점수로 사용한다.

Self-consistency 온도 민감도(HotpotQA seed42, subset120, `n=5`)에서는 `T=0.3→0.7` 구간에서 다양성/분류성이 개선되었고(`Diversity: 0.3150→0.4900`, `AUROC: 0.5731→0.6210`), `T=1.0`에서 AUROC 최고값(`0.6577`)이 관측되었다(표 A.8). 본문 기본값은 과도한 랜덤성을 피하는 보수적 운영 설정으로 `T=0.7`을 유지한다.

본 연구의 메인 결론은 heuristic/self-consistency 중심으로 기술한다. autorater는 본문 핵심 성분이 아니라 semantic confidence reference로 위치를 조정하며, 높은 variance와 parsing noise 가능성을 명시하고 상세 분석은 부록으로 이동한다.

### 4.5 선택적 RAG 정책 모델
정책 집합은 다음과 같다.
1. Baseline: 체커 없이 즉시 생성.
2. Abstain: `INSUFFICIENT`이면 “모르겠습니다.” 반환.
3. Reretrieve: `INSUFFICIENT`이면 `k=3→6→9`로 단계 확장하고 단계별 CSC 재평가 후 재생성.
4. Hybrid(선택): 단계적 재검색/재평가 후에도 불충분이면 abstain.

### 4.6 실험 설계
데이터셋은 HotpotQA/2WikiMultiHopQA를 메인 본문 결과로 사용한다. 메인 비교는 시드 `42~51`의 10-seed, 시드당 subset 120(각 데이터셋 총 1200 샘플)로 수행한다. Natural Questions는 파싱 보정 완료 전까지 부록(provisional)으로 분리한다. 보조 분석 중 핵심 검증 축(calibration, ROC/PR, AURC, latency, `tau` sweep, self-consistency `n=3 vs 5`)은 메인과 동일한 10-seed로 확장해 재집계했다. 평가지표는 EM, F1, 환각률, 커버리지, 선택적 정확도, 지연 시간이며, 추가로 CSC 자체의 분류/보정 품질을 측정한다. Self-consistency는 불확실성 추정 안정화를 위해 `n_samples \ge 5`로 설정한다. 기본 검색 깊이는 `k=3`으로 고정하되, `k` 확장과 feature/temperature 민감도는 탐색적 부록 실험으로 분리 보고한다.  
주2: 본문 기본 생성기/autorater는 모두 `google/flan-t5-large`(약 780M) 계열을 사용했고, 디버깅용 `flan-t5-base`는 메인 표 집계에서 제외했다.

#### CSC 고유 평가 항목
1. Calibration: temperature scaling 전/후 reliability diagram, ECE, Brier.
2. Classification: ROC curve, PR curve, AUROC, AUPRC.
3. Selective risk: Risk-Coverage curve, AURC.
4. Relevance-Answerability 분리: retrieval 점수 분위수별 충분/불충분 비율, Pearson r, Spearman ρ.
5. CSC score vs 최종 정답 여부 상관: Pearson r, Spearman ρ.
6. 정보이론 분석: `I(Retrieval score; Sufficiency)`.
7. 지연 분석: warm-up 제외 평균, latency histogram, CPU/MPS 분리.
8. 정책 최적화: threshold `\tau` sweep 및 비용함수 최소화.

#### 데이터 품질 검증 게이트(재현성/방어력 확보)
1. `gold_answer` 비어있음 비율을 데이터셋별로 보고한다.
2. `oracle_answerable` 양성 비율이 0 또는 1에 붕괴되는지 점검한다.
3. autorater JSON 파싱 성공률을 보고하고, 실패는 `INSUFFICIENT`로 처리한다.
4. 위 조건 위반 시 해당 데이터셋 결과를 본문 메인 결론에서 제외하고 `provisional`로 표기한다.

### 4.7 결과 및 고찰
표 1은 baseline 대비 방법별 평균 성능(10-seed 평균±표준편차, 95% CI)을 제시한다. 표 2~7은 모두 10-seed로 재집계한 보조 분석이며, 표 2는 calibration/ROC, 표 3은 sufficiency와 retrieval quality 분리, 표 4는 Risk-Coverage/AURC, 표 5는 `tau` sweep, 표 6~7은 self-consistency 안정성 및 지연 분석을 다룬다. 추가로 Definition 2 기반 NLI 교차검증(표 8)과 Dense Retriever 일반화(표 9)를 보고한다.

결과 서술 순서는 다음과 같이 고정한다.
1. CSC 정의에 대응하는 calibration/ROC 분석.
2. 선택적 예측 관점의 Risk-Coverage/AURC 분석.
3. Retrieval 점수와 Sufficiency 신호의 분리 분석.
4. 정책 최적화(τ sweep) 및 지연 비용 분석.

#### 핵심 해석 문장(삽입용)
1. CSC-abstain은 환각률을 일관되게 낮추었다(HotpotQA: `Δ=-0.0433`, 95% CI `[-0.0558, -0.0300]`; 2Wiki: `Δ=-0.1042`, 95% CI `[-0.1192, -0.0883]`).
2. CSC-reretrieve는 일부 정확도 개선을 보였으나 지연 비용이 증가하였다.
3. Risk-Coverage 곡선과 AURC 분석에서 환각 감소 효과가 단순 coverage 축소만으로 설명되지 않음을 확인하였다.
4. retrieval 점수와 sufficiency 점수의 상관계수는 전반적으로 낮거나 중간 수준으로, 두 신호가 완전히 중복되지 않음을 시사한다.
주: 10-seed 메인 비교에서 F1 차이 p값은 Hotpot/2Wiki 모두 `0.0001999600079984003`(집계 로그 기준)으로 계산되었다.

#### 현재 실행 결과 요약(2026-03-01 기준)
1. HotpotQA(10-seed, subset120/seed): `Heuristic+Abstain`은 baseline 대비 환각률을 `0.6292 → 0.5858`로 낮추는 대신 F1은 `0.4305 → 0.4143`으로 소폭 하락하였다.
2. 2WikiMultiHopQA(10-seed, subset120/seed): `Heuristic+Abstain`은 환각률을 `0.8275 → 0.7233`으로 낮췄고, F1은 `0.2018 → 0.1916`으로 하락하였다.
3. Autorater는 완화 파서 기준 AUROC가 랜덤 수준(`0.4944`)에 근접하고, strict one-line JSON 재파싱 성공률이 `0.0%`(`N=1500`, `outputs/autorater_failure_reanalysis_hotpot.csv`)로 확인되어 본문 결론에서 제외하고 부록으로 이동한다.
4. Definition 2(NLI entailment) 교차검증에서 QA 성능(F1/환각률/커버리지)은 유지되었으나, calibration 관련 지표(`ECE_after=0.5622`, `AURC=0.8856`)가 `gold_containment` 대비 크게 악화되어, 본문 기본 평가지표는 `gold_containment`를 유지했다.
5. Dense retriever 일반화(Hotpot, 10-seed)에서는 `colbertv2_approx`가 중간 수준 성능(`F1=0.2212`, `Hall=0.5067`)을 보였고, `dpr_nq`는 도메인 불일치로 coverage가 붕괴(`0.0008`)했다.

#### NQ 파싱 보정 후 부록 결과(2026-02-27)
NQ는 `annotations(dict-of-lists)` 파싱 보정과 `drop_empty_gold_answers=true` 적용 후 재실행하였다. 본문 메인 결론은 여전히 HotpotQA/2Wiki 중심으로 유지하되, NQ 보정 결과는 부록 비교표로 보고한다.
| NQ(3-seed, subset120) | F1 | 환각률 | 커버리지 | AURC | 비고 |
|---|---:|---:|---:|---:|---|
| Baseline | 0.0595 ± 0.0448 | 0.5105 ± 0.0931 | 0.5226 ± 0.1055 | - | 파싱 보정 후 기준선 |
| Heuristic+Abstain | 0.0508 ± 0.0341 | 0.4233 ± 0.0677 | 0.4354 ± 0.0798 | 0.3759 | 환각률 감소, 커버리지 감소 |

NQ에서 보정 후에도 F1이 낮은 해석 방향은 다음으로 고정한다.
1. 데이터 성격 차이: Hotpot/2Wiki는 멀티홉 근거 결합형이며, NQ는 오픈도메인 단답형이라 동일 청크 검색 설정(top-k=3,6)에서 답변 문자열 정합이 더 어렵다.
2. 로컬 제약 영향: 본 실험은 소형 로컬 생성기와 경량 검색기를 사용하므로, NQ의 표현 다양성(동의표현/정규화 차이)에 취약하다.
3. 정책 효과 해석: NQ에서는 절대 F1보다 “환각률 감소(0.5105→0.4233)와 커버리지 하락의 trade-off”를 중심으로 해석한다.

따라서 NQ 결과는 “메인 성능 지표”가 아니라 “도메인 이동(open-domain) 스트레스 테스트”로 위치시킨다. 메인 주장(프레임워크 유효성)은 Hotpot/2Wiki에 두고, NQ는 일반화 경향을 보조적으로 보고한다.

### 4.8 결론
본 연구는 CSC를 확률적 answerability 추정 문제로 정식화하고, 선택적 RAG 정책으로 연결하는 통합 프레임워크를 제시하였다. 실험 결과는 환각 감소와 응답 가용성 사이의 구조적 상충관계를 확인하였다. 또한 calibration/ROC 분석을 통해 CSC가 answerability에 대한 근사적 확률 점수를 제공하며, calibration 개선 시 신뢰도 지표로 활용 가능함을 보였다.

#### 결론에 반드시 포함할 전략 문단
본 연구는 환각 감소가 정확도 개선과 동일하지 않음을 실증하였다. 이는 RAG 평가에서 EM/F1 중심 지표가 신뢰도 문제를 충분히 반영하지 못함을 시사한다. 따라서 향후 RAG 시스템은 정확도 최적화가 아니라 위험 제어(risk control) 관점에서 설계되어야 한다.

## 5. 제출 표/그림 구성

### 표 1. 메인 성능 (Hotpot + 2Wiki, 10-seed)
| Dataset | Baseline F1 | Baseline F1 95% CI | 제안방법 F1 | 제안방법 F1 95% CI | Baseline Hallucination | Baseline Hallucination 95% CI | 제안방법 Hallucination | 제안방법 Hallucination 95% CI | Coverage | Coverage 95% CI | Latency(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA (10-seed) | 0.4305 ± 0.0539 | [0.3992, 0.4608] | 0.4143 ± 0.0537 (Heuristic+Abstain) | [0.3819, 0.4437] | 0.6292 ± 0.0579 | [0.5991, 0.6634] | 0.5858 ± 0.0497 | [0.5583, 0.6142] | 0.9158 ± 0.0276 | [0.8991, 0.9308] | 318.6 |
| 2WikiMultiHopQA (10-seed) | 0.2018 ± 0.0285 | [0.1856, 0.2191] | 0.1916 ± 0.0266 (Heuristic+Abstain) | [0.1773, 0.2070] | 0.8275 ± 0.0401 | [0.8017, 0.8484] | 0.7233 ± 0.0291 | [0.7058, 0.7392] | 0.8675 ± 0.0244 | [0.8525, 0.8808] | 594.0 |

주: Table 1의 95% CI는 10-seed(`42~51`) 집계에서 부트스트랩으로 계산하였다.
주2: Table 1의 latency는 warm-up 제외 평균(`outputs/ablations_seed10_hotpot_summary.csv`, `outputs/ablations_seed10_2wiki_summary.csv`) 기준이다.

### 표 2. Calibration/ROC 비교 (10-seed)
| Dataset/Method | ECE(before) | ECE(after) | Brier(before) | Brier(after) | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA / Heuristic | 0.1105 | 0.1075 | 0.2172 | 0.2175 | 0.6337 | 0.7859 |
| HotpotQA / Self-consistency | 0.1800 | 0.1744 | 0.2590 | 0.2493 | 0.5699 | 0.7582 |
| 2Wiki / Heuristic | 0.2755 | 0.1698 | 0.2987 | 0.2691 | 0.6506 | 0.5820 |
| 2Wiki / Self-consistency | 0.2105 | 0.1510 | 0.2977 | 0.2785 | 0.5867 | 0.5644 |

주: Table 2는 시드 `42~51` 집계값(10-seed) 기준이다.
주2: Calibration은 일괄 개선이 아니다. HotpotQA/Heuristic에서는 ECE가 소폭 개선되지만(`0.1105→0.1075`), Brier는 거의 동일/소폭 악화(`0.2172→0.2175`)되어, post-hoc scaling은 데이터·점수 분포 의존적 절차로 해석해야 한다.

### 표 3. Sufficiency vs Retrieval 분리 분석 (10-seed)
| Dataset/Method | 고검색-불충분비율 | 저검색-충분비율 | Pearson r | Spearman ρ | MI |
|---|---:|---:|---:|---:|---:|
| HotpotQA / Heuristic | 0.0292 | 0.8917 | 0.1529 | 0.1722 | 0.3797 |
| HotpotQA / Self-consistency | 0.2500 | 0.7083 | 0.0427 | 0.0449 | 0.1155 |
| 2Wiki / Heuristic | 0.1583 | 0.9292 | -0.0734 | -0.1449 | 0.3563 |
| 2Wiki / Self-consistency | 0.6167 | 0.6375 | -0.1580 | -0.2020 | 0.1529 |

주: 상관계수는 낮거나 중간 수준이며, sufficiency와 retrieval 점수가 완전히 중복되지 않음을 시사한다.

### 표 4. Risk-Coverage 보조 비교 (10-seed)
| Dataset | Method | AURC↓ | ΔAURC vs Heuristic | ΔF1 vs Baseline | ΔHallucination vs Baseline | Latency(ms, warmup 제외) |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Heuristic + Abstain | 0.2252 | 0.0000 | -0.0162 | -0.0433 | 318.6 |
| HotpotQA | Heuristic + Reretrieve | 0.2347 | +0.0095 | +0.0035 | +0.0008 | 364.5 |
| HotpotQA | Self-consistency + Abstain | 0.2449 | +0.0197 | -0.0506 | -0.2092 | 1848.7 |
| HotpotQA | Uncertainty(avg prob) | 0.2425 | +0.0173 | +0.0000 | +0.0000 | 346.4 |
| HotpotQA | Uncertainty(entropy) | 0.2322 | +0.0070 | +0.0000 | +0.0000 | 347.9 |
| 2Wiki | Heuristic + Abstain | 0.4328 | 0.0000 | -0.0102 | -0.1042 | 594.0 |
| 2Wiki | Heuristic + Reretrieve | 0.4330 | +0.0002 | -0.0004 | -0.0008 | 706.4 |
| 2Wiki | Self-consistency + Abstain | 0.4522 | +0.0194 | -0.0587 | -0.4733 | 3555.7 |
| 2Wiki | Uncertainty(avg prob) | 0.4578 | +0.0250 | +0.0000 | +0.0000 | 644.5 |
| 2Wiki | Uncertainty(entropy) | 0.4435 | +0.0107 | +0.0000 | +0.0000 | 675.7 |

주: Heuristic 대비 AURC 차이는 Hotpot에서 Self-consistency가 열세(`+0.0197`), 2Wiki에서도 열세(`+0.0194`)로 관측되었다.

### 표 5. τ Sweep (10-seed)
| Dataset | τ | F1 | Hallucination | AURC |
|---|---:|---:|---:|---:|
| HotpotQA | 0.20 | 0.4305 ± 0.0539 | 0.6292 ± 0.0579 | 0.2252 |
| HotpotQA | 0.35 | 0.4276 ± 0.0545 | 0.6200 ± 0.0552 | 0.2252 |
| HotpotQA | 0.50 | 0.4143 ± 0.0537 | 0.5858 ± 0.0497 | 0.2252 |
| HotpotQA | 0.65 | 0.3339 ± 0.0541 | 0.4442 ± 0.0420 | 0.2252 |
| 2Wiki | 0.20 | 0.2005 ± 0.0284 | 0.7892 ± 0.0387 | 0.4328 |
| 2Wiki | 0.35 | 0.1995 ± 0.0278 | 0.7758 ± 0.0361 | 0.4328 |
| 2Wiki | 0.50 | 0.1916 ± 0.0266 | 0.7233 ± 0.0291 | 0.4328 |
| 2Wiki | 0.65 | 0.1527 ± 0.0264 | 0.5150 ± 0.0415 | 0.4328 |

주: 정책 최적화 비용함수(`L = w_h*Hallucination + w_a*Abstain + w_l*Latency`) 및 최적 `\tau^*`는 부록 표로 분리 보고한다.
주2: AURC는 `checker_score` 순위 기반 지표이므로, 동일 점수 순위에서 임계값 `\tau`만 바꾼 경우 값이 동일하게 유지될 수 있다.
주3: 운영 권장 임계값은 `\tau^*=0.65`(표 A.5)이나, 표 1/4 메인 비교는 방법 간 공정성을 위해 `\tau=0.50` 고정 결과를 사용한다.

### 표 6. Self-consistency 안정성 (n=3 vs n=5, 10-seed)
| Dataset | n_samples | F1 | Hallucination | Coverage | Checker score std | Disagreement mean | Disagreement std |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 3 | 0.3814 | 0.4850 | 0.7992 | 0.2487 | 0.2486 | 0.2487 |
| HotpotQA | 5 | 0.3799 | 0.4200 | 0.7342 | 0.2484 | 0.2972 | 0.2484 |
| 2Wiki | 3 | 0.1412 | 0.4308 | 0.5508 | 0.2701 | 0.3942 | 0.2701 |
| 2Wiki | 5 | 0.1431 | 0.3542 | 0.4817 | 0.2788 | 0.4663 | 0.2788 |

주: `n=5`는 절대 성능 우위보다 추정 안정성/불일치 신호 분해능 확보 관점에서 채택했다.

### 표 7. 지연 분석 (10-seed)
| Dataset/Method | warm-up 제거 평균(ms) | 전체지연 평균 95% CI | 지연 표준편차(ms) | CPU 평균(ms) | MPS 평균(ms) |
|---|---:|---:|---:|---:|---:|
| HotpotQA / Baseline | 357.6 | [346.5, 377.0] | 260.0 | - | 361.0 |
| HotpotQA / Heuristic + Abstain | 318.6 | [304.3, 332.4] | 237.3 | - | 318.1 |
| HotpotQA / Self-consistency + Abstain | 1848.7 | [1787.3, 1902.4] | 970.9 | - | 1844.2 |
| HotpotQA / Uncertainty(avg prob) | 346.4 | [330.2, 359.8] | 251.5 | - | 344.8 |
| HotpotQA / Uncertainty(entropy) | 347.9 | [332.8, 361.8] | 240.3 | - | 346.9 |
| 2Wiki / Baseline | 649.7 | [629.4, 677.9] | 403.3 | - | 653.0 |
| 2Wiki / Heuristic + Abstain | 594.0 | [573.3, 624.9] | 426.9 | - | 598.3 |
| 2Wiki / Self-consistency + Abstain | 3555.7 | [3476.5, 3675.6] | 1644.2 | - | 3572.1 |
| 2Wiki / Uncertainty(avg prob) | 644.5 | [622.9, 669.1] | 382.8 | - | 645.5 |
| 2Wiki / Uncertainty(entropy) | 675.7 | [654.5, 705.1] | 404.3 | - | 680.0 |

주: 본 10-seed 실행은 기록상 `mps` 장치에서 수행되었고 CPU 경로는 관측되지 않아 CPU 평균은 `-`로 표기한다.

### 표 8. Answerable 정의 교차검증 (HotpotQA, 10-seed)
| Answerable 정의 | 메서드 | F1 | 환각률 | 커버리지 | AURC | CSC_AUROC | CSC_ECE(after) |
|---|---|---:|---:|---:|---:|---:|---:|
| gold_containment | Heuristic + Abstain | 0.4143 | 0.5858 | 0.9158 | 0.2252 | 0.6337 | 0.1075 |
| entailment | Heuristic + Abstain | 0.4143 | 0.5858 | 0.9158 | 0.8856 | 0.6675 | 0.5622 |

주: 최종 QA 지표는 동일했지만, score 스케일과 calibration(ECE, AURC)은 정의에 따라 크게 달라졌다. 따라서 Definition 2(NLI)는 본문 주장에 대한 교차검증 신호로 사용한다.

### 표 9. 검색기 일반화 검증 (HotpotQA, 10-seed)
| 리트리버 | 메서드 | F1 | 환각률 | 커버리지 | AURC | CSC_AUROC | 체커파싱성공률 |
|---|---|---:|---:|---:|---:|---:|---:|
| e5_small | Heuristic + Abstain | 0.4143 | 0.5858 | 0.9158 | 0.2252 | 0.6337 | 1.0000 |
| dpr_nq | Heuristic + Abstain | 0.0008 | 0.0000 | 0.0008 | 0.9708 | 0.4031 | 1.0000 |
| colbertv2_approx | Heuristic + Abstain | 0.2212 | 0.5067 | 0.6692 | 0.4652 | 0.6392 | 1.0000 |

주: Dense retriever에서도 CSC 파이프라인은 동작했지만, 리트리버 품질 차이가 end-to-end 성능을 지배했다. 특히 `dpr_nq`는 도메인 불일치로 coverage가 붕괴했다.

## 7.4 Reretrieve-Hybrid Trade-off
본 절 수치는 표 4(10-seed 보조 비교) 기준이다.
추가 실험(Heuristic 기준)에서 Reretrieve는 Abstain 대비 F1 회복에 기여했다. HotpotQA는 `0.4143→0.4340`(`+0.0197`), 2Wiki는 `0.1916→0.2015`(`+0.0098`)였다.

다만 환각률은 동일 비교에서 악화되었다. HotpotQA는 `0.5858→0.6300`(`+0.0442`), 2Wiki는 `0.7233→0.8267`(`+0.1033`)으로 증가했다. 즉, 본 설정에서는 Reretrieve가 정확도 회복에는 유효하지만 환각 억제와는 충돌한다.

지연 비용도 증가했다. Reretrieve는 Heuristic+Abstain 대비 Hotpot에서 약 `+45.9ms`, 2Wiki에서 약 `+112.4ms`가 추가되었다.

종합하면, 본 실험에서는 Reretrieve가 coverage/F1 회복의 대안이지만, BM25 기반 재검색만으로 환각을 일관되게 줄이기는 어려웠다.

## 7.5 k-Sweep 확장 분석 (Heuristic + Abstain, 3-seed)
`k={3,5,7}` 확장 실험에서 두 데이터셋 모두 `k` 증가에 따라 coverage가 상승했다(Hotpot: `0.9067→0.9500→0.9680`, 2Wiki: `0.8589→0.9089→0.9322`).  
동시에 hallucination도 함께 증가했다(Hotpot: `0.5973→0.6073→0.6213`, 2Wiki: `0.7311→0.7711→0.7956`).

F1은 Hotpot에서 `k` 증가에 따라 상승(`0.4021→0.4388→0.4448`)했고, 2Wiki는 `k=5`에서 최대(`0.1962`) 후 `k=7`에서 소폭 하락(`0.1918`)했다.  
지연 시간은 단조 증가했다(Hotpot: `648.1→898.6→988.8ms`, 2Wiki: `961.9→1335.7→1518.6ms`).

즉, `k=3→6`류의 검색 확장은 coverage/F1 회복 잠재력이 있으나, 환각 및 지연 증가 비용이 동반되는 경향을 확인했다. 본 연구는 기본 정책의 목표를 “환각 위험 제어”에 두므로 `k=3`을 기본값으로 유지하고, 재검색은 필요 시 선택적으로 활성화하는 설정이 타당하다.

## 7.6 Limitations
메인 비교와 핵심 보조분석(표 2~7, NLI 교차검증, retriever 일반화)은 `seed=42~51`의 10-seed 반복으로 확장했다. 다만 일부 탐색적 부록 실험(예: feature ablation, temperature sensitivity, 일부 k-sweep)은 여전히 3-seed 또는 단일-seed 기반이다. paired t-test(양측, `alpha=0.05`, power=0.8) 기준 최소 검출 효과크기(대응표본 Cohen's `d_z`)는 `n=10`에서 약 `0.996`, `n=3`에서 약 `3.26`으로 차이가 크다.
따라서 본문 핵심 결론은 10-seed 근거로 해석하되, 부록 탐색 결과는 방향성 증거로 제한적으로 해석해야 한다.

### 부록 A. Autorater 및 Natural Questions 결과

#### 표 A.5. 정책 비용 최적화(Heuristic, 3-seed 집계)
| Dataset | τ | L(w_h=1,w_a=0.5,w_l=0.001) | Hallucination | AbstainRate | Latency(ms) |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.20 | 1.0983 | 0.6387 | 0.0407 | 439.3 |
| HotpotQA | 0.35 | 1.0851 | 0.6300 | 0.0513 | 429.4 |
| HotpotQA | 0.50 | 1.0582 | 0.5973 | 0.0933 | 414.2 |
| HotpotQA | 0.65 | 0.9126 | 0.4480 | 0.2913 | 318.9 |
| 2WikiMultiHopQA | 0.20 | 1.7765 | 0.7989 | 0.0689 | 943.2 |
| 2WikiMultiHopQA | 0.35 | 1.7246 | 0.7844 | 0.0833 | 898.5 |
| 2WikiMultiHopQA | 0.50 | 1.6751 | 0.7311 | 0.1411 | 873.5 |
| 2WikiMultiHopQA | 0.65 | 1.3766 | 0.5267 | 0.3722 | 663.8 |

주: `L = w_h*Hallucination + w_a*AbstainRate + w_l*Latency(ms)`로 계산했으며, 기본 가중치(`w_h=1.0, w_a=0.5, w_l=0.001`)에서 두 데이터셋 모두 최적 임계값은 `τ*=0.65`였다.
주2: 가중치 민감도(`w_h`를 `0.5~2.0`으로 변경) 분석에서도 두 데이터셋의 `τ*`는 모두 `0.65`로 유지되어, 본 탐색 구간에서는 hallucination 비용 가중 증가가 `τ*`를 추가 이동시키지 않았다.
주3: 다만 메인 본문 표(표 1 및 표 4/5 보조 비교)는 방법 간 공정 비교를 위해 `τ=0.50` 고정 설정을 유지하고, `τ*=0.65`는 운영 권장값으로 분리 보고한다.
주4: Coverage는 `1 - AbstainRate`로 환산되며, `τ` 증가에 따라 AbstainRate가 커질수록 Coverage는 단조 감소한다.

#### 표 A.6. k-Sweep (Heuristic+Abstain, 3-seed)
| Dataset | k | F1 | Hallucination | Coverage | Latency(ms) |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 3 | 0.4021 ± 0.0248 | 0.5973 ± 0.0391 | 0.9067 ± 0.0133 | 648.1 ± 51.4 |
| HotpotQA | 5 | 0.4388 ± 0.0041 | 0.6073 ± 0.0110 | 0.9500 ± 0.0053 | 898.6 ± 25.2 |
| HotpotQA | 7 | 0.4448 ± 0.0175 | 0.6213 ± 0.0110 | 0.9680 ± 0.0020 | 988.8 ± 65.2 |
| 2WikiMultiHopQA | 3 | 0.1824 ± 0.0043 | 0.7311 ± 0.0267 | 0.8589 ± 0.0201 | 961.9 ± 172.0 |
| 2WikiMultiHopQA | 5 | 0.1962 ± 0.0042 | 0.7711 ± 0.0252 | 0.9089 ± 0.0184 | 1335.7 ± 241.4 |
| 2WikiMultiHopQA | 7 | 0.1918 ± 0.0100 | 0.7956 ± 0.0184 | 0.9322 ± 0.0192 | 1518.6 ± 213.2 |

주: 본 표는 `outputs/ablation_v8_k_sweep_heuristic_abstain_summary.csv`(seed `42/43/44`) 집계값이다.
주2: Table 1 latency와 수치가 다른 이유는 실행 배치가 다르기 때문이다(Table 1: `ablation_v6` 메인 비교, 표 A.6: `ablation_v8` k-sweep 전용 실행). 따라서 절대 지연값의 교차표 비교보다, 각 표 내부의 `k` 변화 추세 해석을 우선한다.

#### 표 A.7. Heuristic Feature Ablation (`g_H1/g_H2/g_H3`, 3-seed)
| Estimator variant | AUROC(Hotpot) | AUROC(2Wiki) | Latency추가(ms) |
|---|---:|---:|---:|
| g_H1 (binary token overlap) | 0.6283 ± 0.0076 | 0.6507 ± 0.0214 | Hotpot +0.000 / 2Wiki +0.000 |
| g_H2 (TF-IDF weighted coverage) | 0.6383 ± 0.0072 | 0.6582 ± 0.0306 | Hotpot +0.018 / 2Wiki +0.018 |
| g_H3 (Named Entity match ratio) | 0.5285 ± 0.0059 | 0.5677 ± 0.0299 | Hotpot +0.046 / 2Wiki +0.051 |

주: `g_H2`는 `score = Σ(tfidf(t), t∈q∩ctx) / Σ(tfidf(t), t∈q)`로 계산했다. `g_H3`는 규칙 기반 개체 추출 후 `|NE(q)∩NE(ctx)| / max(|NE(q)|,1)`를 사용했다.
주2: 결과는 `outputs/heuristic_variant_ablation_v1_table.md` 기준이며, 지연은 checker 호출 추가 비용(평균 ms)이다.
주3: `g_H2`는 AUROC 기준으로 `g_H1` 대비 Hotpot `+0.010p`, 2Wiki `+0.008p`의 소폭 개선이 있으나, corpus-level IDF 사전 관리 없이도 경쟁력 있는 성능을 제공하는 단순성/재현성 측면에서 본문 기본 추정기는 `g_H1`을 유지한다.

#### 표 A.8. Self-consistency Temperature 민감도 (HotpotQA, seed42, subset120, n=5)
| T | Diversity | Var(g_SC) | AUROC |
|---:|---:|---:|---:|
| 0.3 | 0.3150 | 0.039258 | 0.5731 |
| 0.5 | 0.3983 | 0.056176 | 0.5658 |
| 0.7 | 0.4900 | 0.071034 | 0.6210 |
| 1.0 | 0.6033 | 0.071639 | 0.6577 |

주: Diversity는 `unique answers / n`의 샘플 평균값이다. `Var(g_SC)`는 샘플별 self-consistency 점수 분산이다.
주2: 결과 파일은 `outputs/self_consistency_temp_sensitivity_hotpot_seed42.csv` 및 `outputs/self_consistency_temp_sensitivity_hotpot_seed42.md`이다.

#### 표 A.9. Autorater 실패 정량 분석 (HotpotQA, `ablations_v3` seed42/43/44 재분석)
| 항목 | 값 |
|---|---:|
| 총 샘플 수 | 1500 |
| strict one-line JSON 성공률 | 0.0000 |
| 완화 파서 AUROC | 0.4944 |
| 파싱 방식 분포(`json_fragment`) | 97.07% |
| 파싱 방식 분포(`label_only_recovery`) | 2.33% |
| 원래부터 strict JSON 유효 | 0.60% |
| confidence mass in [0.8, 1.0] | 90.2% |

주: 본 표는 `outputs/autorater_failure_reanalysis_hotpot.csv`에 기반한다. 즉, strict JSON 강제 조건에서는 실사용 가능한 파싱률이 사실상 0에 수렴하며, 완화 파서에서도 분별력은 랜덤 수준에 가깝다.

1. Autorater 본문 제외 사유: AUROC가 랜덤 수준에 근접하고, calibration error가 높아 메인 결론을 약화시킴.
2. NQ 파싱 보정: `annotations` 포맷(dict-of-lists) 대응, yes/no(0/1) 매핑, short/long answer fallback, 빈 정답 샘플 제외 옵션 적용.
3. NQ 재실행 산출물: `outputs/nq_parsefix_v2_subset120_summary.csv`, `outputs/nq_parsefix_v2_subset120_summary.md`.

### 그림 구성
1. 시스템 개요도: Query → Retrieve → CSC → Generate/Reretrieve/Abstain  
2. Reliability diagram (CSC score vs 실제 정답 확률)  
3. ROC / PR curves  
4. CSC score density plot (SUFFICIENT vs INSUFFICIENT, threshold 표시)
5. Risk-Coverage curve + AURC
6. Retrieval score vs sufficiency probability scatter plot + decision boundary
7. Latency histogram (warm-up 제거 전/후 요약 포함)
8. k-sweep 3D trade-off (F1, Hallucination, Latency)
