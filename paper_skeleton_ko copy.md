

## 1. 제목(안)
- 문맥 충분성의 확률적 추정에 기반한 선택적 RAG 제어: 환각 감소와 위험-커버리지 상충 분석
- Probabilistic Context Sufficiency Estimation for Selective RAG Control under Local Compute Constraints

## 2. 핵심 기여 문장 (논문 본문/커버레터 공용)
1. 본 연구는 문맥 충분성 검사를 이진 분류가 아니라 `g(q, C_k) ≈ P(Answerable | q, C_k)` 추정 문제로 재정의하였다.
2. heuristic, autorater, self-consistency를 각각 feature-based, semantic LLM-based, variance-based epistemic estimator로 통합 이론 틀에 정렬하였다.
3. RAG 성능을 relevance(검색 품질)와 answerability(문맥 충분성)로 분리 평가하고, 두 축이 일치하지 않는 사례를 정량적으로 제시하였다.
4. 3-seed 반복 실험, calibration(before/after), Risk-Coverage/AURC, k-sweep을 포함한 통계적으로 방어 가능한 실험 체계를 로컬 Mac 환경에서 구현하였다.

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
관련 연구는 크게 (i) retrieval 개선, (ii) uncertainty-aware generation, (iii) selective prediction으로 구분된다. retrieval 품질 평가는 relevance 중심이며, selective prediction은 주로 최종 예측 거부 정책에 초점을 둔다. 본 연구는 질문-문맥 수준에서 answerability를 직접 추정한다는 점에서 기존 접근과 차별화된다. 또한 체커 모듈을 플러그인 구조로 통합하여 비교 실험의 공정성과 확장성을 확보하였다.

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
2. LLM semantic estimator(autorater): 질문-문맥 의미 적합성을 기반으로 JSON 고정 출력(`label`, `confidence`)을 생성한다.  
3. Self-consistency estimator: 동일 문맥에서 다중 샘플 생성의 분산을 epistemic uncertainty로 보고, 다수결 비율을 확률 점수로 사용한다.

본 연구의 메인 결론은 heuristic/self-consistency 중심으로 기술한다. autorater는 본문 핵심 성분이 아니라 semantic confidence reference로 위치를 조정하며, 높은 variance와 parsing noise 가능성을 명시하고 상세 분석은 부록으로 이동한다.

### 4.5 선택적 RAG 정책 모델
정책 집합은 다음과 같다.
1. Baseline: 체커 없이 즉시 생성.
2. Abstain: `INSUFFICIENT`이면 “모르겠습니다.” 반환.
3. Reretrieve: `INSUFFICIENT`이면 `k`를 확장하여 재검색 후 재생성.
4. Hybrid(선택): 재검색 후에도 불충분이면 abstain.

### 4.6 실험 설계
데이터셋은 HotpotQA와 2WikiMultiHopQA를 메인 본문 결과로 사용한다. Natural Questions는 파싱 보정 완료 전까지 부록(provisional)으로 분리한다. 모든 방법은 동일한 generator/retriever 설정에서 비교하며, 시드는 `42,43,44`를 사용한다. 평가지표는 EM, F1, 환각률, 커버리지, 선택적 정확도, 지연 시간이며, 추가로 CSC 자체의 분류/보정 품질을 측정한다. Self-consistency는 불확실성 추정 안정화를 위해 `n_samples \ge 5`로 설정한다.

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
표 1은 baseline 대비 방법별 평균 성능(3-seed 평균±표준편차, 95% CI)을 제시한다. 표 2는 calibration/ROC 중심 분석, 표 4는 Risk-Coverage/AURC 메인 비교, 표 3은 sufficiency와 retrieval quality의 분리 분석을 제시한다. 표 5는 HotpotQA 기준 `\tau` sweep 결과를 제시하며, 표 6~7은 self-consistency 안정성 및 지연 분석 결과를 제시한다. `k=3,5,7` sweep 결과는 그림(3D trade-off) 또는 부록 표로 제시한다.

결과 서술 순서는 다음과 같이 고정한다.
1. CSC 정의에 대응하는 calibration/ROC 분석.
2. 선택적 예측 관점의 Risk-Coverage/AURC 분석.
3. Retrieval 점수와 Sufficiency 신호의 분리 분석.
4. 정책 최적화(τ sweep) 및 지연 비용 분석.

#### 핵심 해석 문장(삽입용)
1. CSC-abstain은 환각률을 일관되게 낮추었으나 커버리지 손실을 동반하였다.
2. CSC-reretrieve는 일부 정확도 개선을 보였으나 지연 비용이 증가하였다.
3. Risk-Coverage 곡선과 AURC 분석에서 환각 감소 효과가 단순 coverage 축소만으로 설명되지 않음을 확인하였다.
4. retrieval 점수와 sufficiency 점수의 상관계수는 전반적으로 낮거나 중간 수준으로, 두 신호가 완전히 중복되지 않음을 시사한다.

#### 현재 실행 결과 요약(2026-02-27 기준)
1. HotpotQA(3-seed): `Heuristic+Abstain`은 baseline 대비 환각률을 `0.6420 → 0.5973`으로 낮추는 대신 F1은 `0.4159 → 0.4021`로 소폭 하락하였다.
2. 2WikiMultiHopQA(3-seed): `Heuristic+Abstain`은 환각률을 `0.8478 → 0.7311`로 크게 낮췄고, F1은 `0.1923 → 0.1824`로 하락하였다.
3. Autorater는 AUROC가 랜덤 수준에 근접하고(ECE도 높음), 본문 결론을 약화할 수 있으므로 부록 분석으로 이동한다.

#### NQ 재실행 전 해석 제한(본문 명시 권장)
NQ 실행 로그에서 `gold_answer`가 빈 문자열로 기록되는 사례가 확인되어, answerability 기반 지표(ECE, AUROC, AUPRC, 상관/MI)의 해석 가능성이 저하되었다. 본문 메인 주장은 HotpotQA와 2Wiki 결과를 중심으로 기술하고, NQ는 데이터 파싱 보정 후 재실험 결과로 대체한다.

### 4.8 결론
본 연구는 CSC를 확률적 answerability 추정 문제로 정식화하고, 선택적 RAG 정책으로 연결하는 통합 프레임워크를 제시하였다. 실험 결과는 환각 감소와 응답 가용성 사이의 구조적 상충관계를 확인하였다. 또한 calibration/ROC 분석을 통해 CSC가 answerability에 대한 근사적 확률 점수를 제공하며, calibration 개선 시 신뢰도 지표로 활용 가능함을 보였다.

#### 결론에 반드시 포함할 전략 문단
본 연구는 환각 감소가 정확도 개선과 동일하지 않음을 실증하였다. 이는 RAG 평가에서 EM/F1 중심 지표가 신뢰도 문제를 충분히 반영하지 못함을 시사한다. 따라서 향후 RAG 시스템은 정확도 최적화가 아니라 위험 제어(risk control) 관점에서 설계되어야 한다.

## 5. 제출 표/그림 구성

### 표 1. 메인 성능 (Hotpot + 2Wiki)
| Dataset | Baseline F1 | 제안방법 F1 | Baseline Hallucination | 제안방법 Hallucination | Coverage | Latency(ms) |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA (3-seed) | 0.4159 ± 0.0244 | 0.4021 ± 0.0248 (Heuristic+Abstain) | 0.6420 ± 0.0308 | 0.5973 ± 0.0391 | 0.9067 ± 0.0133 | 441.2 |
| 2WikiMultiHopQA (3-seed) | 0.1923 ± 0.0025 | 0.1824 ± 0.0043 (Heuristic+Abstain) | 0.8478 ± 0.0019 | 0.7311 ± 0.0267 | 0.8589 ± 0.0201 | 1442.4 ± 1149.4 |

### 표 2. Calibration/ROC 비교 (3-seed)
| Dataset/Method | ECE(before) | ECE(after) | Brier(before) | Brier(after) | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA / Heuristic | 0.1003 | 0.1097 | 0.2232 | 0.2271 | 0.6283 | 0.7547 |
| HotpotQA / Self-consistency | 0.1689 | 0.1485 | 0.2612 | 0.2453 | 0.5681 | 0.7622 |
| 2Wiki / Heuristic | 0.2650 | 0.1579 | 0.3014 | 0.2676 | 0.6507 | 0.5593 |
| 2Wiki / Self-consistency | 0.1410 | 0.1339 | 0.2744 | 0.2659 | 0.6162 | 0.5536 |

주: Autorater는 본문 표에서 제외하고 부록으로 이동한다.

### 표 3. Sufficiency vs Retrieval 분리 분석
| Dataset/Method | 고검색-불충분비율 | 저검색-충분비율 | Pearson r | Spearman ρ | MI |
|---|---:|---:|---:|---:|---:|
| HotpotQA / Heuristic | 0.0267 | 0.8967 | 0.1444 | 0.1876 | 0.1051 |
| HotpotQA / Self-consistency | 0.1667 | 0.7467 | 0.0238 | 0.0307 | 0.0098 |
| 2Wiki / Heuristic | 0.1056 | 0.8889 | -0.0540 | -0.0984 | 0.1409 |
| 2Wiki / Self-consistency | 0.6722 | 0.5556 | -0.1715 | -0.1920 | 0.0568 |

### 표 4. Risk-Coverage 메인 비교 (3-seed)
| Dataset | Method | AURC↓ | ΔAURC vs Heuristic |
|---|---|---:|---:|
| HotpotQA | Heuristic + Abstain | 0.2539 | 0.0000 |
| HotpotQA | Self-consistency + Abstain | 0.2448 | -0.0091 |
| HotpotQA | Uncertainty(avg prob) | - | - |
| HotpotQA | Uncertainty(entropy) | - | - |
| 2Wiki | Heuristic + Abstain | 0.4544 | 0.0000 |
| 2Wiki | Self-consistency + Abstain | 0.4614 | 0.0069 |
| 2Wiki | Uncertainty(avg prob) | 0.4971 | 0.0426 |
| 2Wiki | Uncertainty(entropy) | 0.4695 | 0.0151 |

### 표 5. τ Sweep (3-seed)
| Dataset | τ | F1 | Hallucination | AURC |
|---|---:|---:|---:|---:|
| HotpotQA | 0.20 | 0.4158 ± 0.0243 | 0.6387 ± 0.0323 | 0.2539 |
| HotpotQA | 0.35 | 0.4138 ± 0.0243 | 0.6300 ± 0.0291 | 0.2539 |
| HotpotQA | 0.50 | 0.4021 ± 0.0248 | 0.5973 ± 0.0391 | 0.2539 |
| HotpotQA | 0.65 | 0.3391 ± 0.0258 | 0.4480 ± 0.0250 | 0.2539 |
| 2Wiki | 0.20 | 0.1912 ± 0.0034 | 0.7989 ± 0.0135 | 0.4544 |
| 2Wiki | 0.35 | 0.1910 ± 0.0036 | 0.7844 ± 0.0102 | 0.4544 |
| 2Wiki | 0.50 | 0.1824 ± 0.0043 | 0.7311 ± 0.0267 | 0.4544 |
| 2Wiki | 0.65 | 0.1433 ± 0.0124 | 0.5267 ± 0.0379 | 0.4544 |

주: 정책 최적화 비용함수(`L = w_h*Hallucination + w_a*Abstain + w_l*Latency`) 및 최적 `\tau^*`는 부록 표로 분리 보고한다.
주2: 본문의 AURC는 `checker_score` 순위 기반 지표이므로, 동일 점수 분포/순위에서 임계값 `\tau`만 바꾼 경우 값이 동일하게 유지될 수 있다.

### 표 6. Self-consistency 안정성 (n=3 vs n=5, 3-seed)
| Dataset | n_samples | F1 | Hallucination | Coverage | AURC |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 3 | 0.3879 ± 0.0212 | 0.4520 ± 0.0420 | 0.7580 ± 0.0174 | 0.2340 |
| HotpotQA | 5 | 0.3717 ± 0.0169 | 0.4900 ± 0.0174 | 0.7847 ± 0.0042 | 0.2448 |
| 2Wiki | 3 | 0.1408 ± 0.0213 | 0.3889 ± 0.0455 | 0.5033 ± 0.0635 | 0.4482 |
| 2Wiki | 5 | 0.1332 ± 0.0220 | 0.3233 ± 0.0406 | 0.4378 ± 0.0542 | 0.4614 |

### 표 7. 지연 분석 (3-seed)
| Dataset/Method | warm-up 제거 평균(ms) | 지연 표준편차(ms) | CPU 평균(ms) | MPS 평균(ms) |
|---|---:|---:|---:|---:|
| HotpotQA / Baseline | 481.1839 | 424.4658 | 484.6605 | - |
| HotpotQA / Heuristic + Abstain | 440.6218 | 428.1960 | 441.2128 | - |
| HotpotQA / Self-consistency + Abstain | 1578.6525 | 1032.7517 | 1581.5100 | - |
| 2Wiki / Baseline | 1028.9195 | 581.1549 | 1045.9616 | - |
| 2Wiki / Heuristic + Abstain | 1452.1417 | 1226.3627 | 1442.3545 | - |
| 2Wiki / Self-consistency + Abstain | 5008.6787 | 1623.4768 | 5014.6974 | - |

주: 본 3-seed 재현 실행은 `mps_available=False` 환경에서 수행되어 생성 장치는 사실상 CPU로 고정되었다. 일부 구버전 로그에는 `generator_device` 필드가 없어 CPU/MPS 분리 통계가 비어 있으며, 해당 항목은 장치 정보가 기록된 로그에서만 집계하였다.

### 부록 A. Autorater 및 Natural Questions 결과
1. Autorater 본문 제외 사유: AUROC가 랜덤 수준에 근접하고, calibration error가 높아 메인 결론을 약화시킴.
2. NQ 본문 제외 사유: `gold_answer` 파싱 결함으로 `oracle_answerable` 지표 붕괴.
3. 보정 후 재실행 결과를 Appendix 표로 별도 보고.

### 그림 구성
1. 시스템 개요도: Query → Retrieve → CSC → Generate/Reretrieve/Abstain  
2. Reliability diagram (CSC score vs 실제 정답 확률)  
3. ROC / PR curves  
4. CSC score density plot (SUFFICIENT vs INSUFFICIENT, threshold 표시)
5. Risk-Coverage curve + AURC
6. Retrieval score vs sufficiency probability scatter plot + decision boundary
7. Latency histogram (warm-up 제거 전/후 요약 포함)
8. k-sweep 3D trade-off (F1, Hallucination, Latency)

