# LLM 바닥부터 만들기 (코딩 중심 커리큘럼)

대상: **수학은 중학교 수준(OK)**, **코딩은 익숙함**  
목표: “개념만”이 아니라 **직접 학습/추론되는 텍스트 생성 모델**을 단계적으로 만든다.

용어가 한 번에 안 잡히면 [GLOSSARY](GLOSSARY.md)를 먼저 보고 오면 좋습니다(용어/shape 위주).

---

## 0) 이 커리큘럼의 원칙

- **수학 최소화 + 코드 최대화**: 필요한 수학은 “왜 필요한지/어디에 쓰는지”만 잡고, 나머지는 실습으로 체득합니다.
- **작게 만들고, 자주 확인**: 작은 데이터/작은 모델로 빠르게 결과를 보고 점진적으로 확장합니다.
- **2트랙 운영**
  - **Core 트랙 (numpy만)**: 설치 부담 없이 “언어모델이 돌아가는 원리”를 끝까지 이해.
  - **LLM 트랙 (PyTorch 권장)**: Transformer(Attention) 기반 “진짜 LLM 구조”로 미니 GPT를 학습/생성.

> 참고: 현재 이 작업 폴더 환경은 `Python 3.14` + `numpy`는 준비되어 있고, `torch`는 설치되어 있지 않습니다.  
> 실습은 **Core 트랙을 먼저 진행**하고, LLM 트랙은 **Python 3.12/3.13 + PyTorch** 환경에서 진행하도록 설계합니다(권장).

---

## 1) 최종 산출물(만들게 될 것)

### Core 트랙 (numpy)
1. **빅램(2-gram) 카운트 모델**: 확률로 다음 글자/토큰 생성
2. **신경망 빅램 모델**: 임베딩 + 소프트맥스로 학습되는 최소 신경망 LM
3. **미니 MLP LM**: 컨텍스트 길이를 늘리고 성능 개선(배치/학습루프/평가 포함)

### LLM 트랙 (선택, PyTorch)
4. **Self-Attention**을 직접 구현(프레임워크의 “Transformer 레이어”는 사용하지 않음)
5. **MiniGPT(작은 Transformer LM)** 학습 + 샘플링(temperature/top-k)
6. (선택) **토크나이저(BPE)**: 서브워드로 시퀀스 길이/희귀 토큰 문제를 다룸
7. (선택) **파인튜닝 입문**: SFT(지도 파인튜닝), LoRA/QLoRA, 평가/배포에서 나오는 필수 용어/체크리스트

---

## 2) 권장 시간표 (8주 + 확장 2주 옵션, 주 5~7시간 기준)

### Week 0 — 준비/감 잡기
- LLM이 하는 일(다음 토큰 예측), 데이터/토큰/모델/학습의 큰 그림
- 딥러닝 기초(코드로): 퍼셉트론(`y=w*x+b`), 행렬곱(`W@x+b`), 경사하강법, ReLU
- 실습: `demo_week0_dl_basics.py` 실행 + “햄버거 가게 행렬곱” 과제

> 📂 레슨: [00_setup](lessons/00_setup.md), [00_dl_basics](lessons/00_dl_basics.md) | 과제: [WEEK0](exercises/WEEK0.md)

### Week 1 — 확률 기반 언어모델 (빅램 카운트)
- 조건부확률 `P(next | prev)` 직관
- 실습: 카운트로 확률표 만들기 + 샘플링으로 문장 생성

> 📂 레슨: [01_why_language_model](lessons/01_why_language_model.md), [02_tokenization_char](lessons/02_tokenization_char.md), [03_bigram_counts](lessons/03_bigram_counts.md) | 과제: [WEEK1](exercises/WEEK1.md)

### Week 2 — 신경망의 최소 구성요소 (임베딩/소프트맥스)
- “단어를 숫자 벡터로” (임베딩) + “확률로” (소프트맥스)
- 실습: **신경망 빅램 LM** 학습(크로스엔트로피 손실, 경사하강)

> 📂 레슨: [04_neural_bigram](lessons/04_neural_bigram.md) | 과제: [WEEK2](exercises/WEEK2.md)

### Week 3 — 학습 루프를 내 손으로 (배치/검증/과적합)
- 데이터 분할(train/val), 배치, 학습률, early stopping 감각
- 실습: 미니 MLP LM로 확장 + 간단 평가(손실/퍼플렉서티)

> 📂 레슨: [05_mlp_context_lm](lessons/05_mlp_context_lm.md) | 과제: [WEEK3](exercises/WEEK3.md)

### Week 4 — Attention 직관 만들기
- “필요한 정보에만 집중”이 왜 유리한가
- 실습: (Core) Attention을 “계산”해보기 / (LLM) PyTorch로 Self-Attention 구현

> 📂 레슨: [06_self_attention](lessons/06_self_attention.md) | 과제: [WEEK4](exercises/WEEK4.md)

### Week 5 — Transformer 블록 조립
- LayerNorm, Residual, FFN, Positional Encoding
- 실습: MiniGPT 모델 조립 + 학습이 실제로 내려가는지 확인

> 📂 레슨: [07_transformer_block](lessons/07_transformer_block.md) | 과제: [WEEK5](exercises/WEEK5.md)

### Week 6 — 생성 품질 올리기 (샘플링/디코딩)
- greedy vs sampling, temperature, top-k/top-p
- 실습: 생성 옵션 비교 + “내 데이터”로 결과 튜닝

> 📂 레슨: [08_sampling_and_eval](lessons/08_sampling_and_eval.md) | 과제: [WEEK6](exercises/WEEK6.md)

### Week 7 — 마무리/확장 로드맵
- 토크나이저(BPE), 더 큰 데이터/모델, 속도/메모리, 평가/안전
- 실습: (선택) BPE 토크나이저 구현 + 내 데이터로 “재학습/도메인 적응” 실험

> 📂 레슨: [09_bpe_tokenizer](lessons/09_bpe_tokenizer.md) | 과제: [WEEK7](exercises/WEEK7.md)

### Week 8 (선택) — 파인튜닝 선수지식(용어/흐름)
- pre-training vs post-training, base model vs fine-tuned model
- SFT(지도 파인튜닝), instruction/chat 데이터, ground truth, eval loss
- prompt engineering vs fine-tuning vs RAG(언제 무엇을 쓰는지)
- 실습: 파인튜닝 데이터 포맷(JSONL) 만들기 + 출력 구조 검증(형식/스키마)

> 📂 레슨: [11_finetuning_essentials](lessons/11_finetuning_essentials.md) | 과제: [WEEK8](exercises/WEEK8.md)

### Week 9 (선택) — LoRA/QLoRA + 운영 감각
- LoRA/QLoRA(왜 필요한지, 무엇을 업데이트하는지)
- VRAM/정밀도(FP16/BF16), gradient accumulation/checkpointing, 시퀀스 길이 튜닝
- 평가/회귀 테스트, 추론 최적화(quantization/distillation, 프롬프트 단순화)

> 📂 레슨: [12_lora_qlora_and_ops](lessons/12_lora_qlora_and_ops.md) | 과제: [WEEK8](exercises/WEEK8.md)

---

## 3) 추천 자료(선택)
“필수”가 아니라, 막힐 때 참고용입니다.

- 유튜브: **3Blue1Brown(선형대수 직관)**, **StatQuest(손실/확률)**, **Andrej Karpathy(Zero to Hero)**
- 블로그: **The Illustrated Transformer (Jay Alammar)**, **Attention Is All You Need 해설 글들**
- 서적:
  - **밑바닥부터 시작하는 딥러닝(Deep Learning from Scratch)**: numpy로 신경망 감 잡기
  - **Dive into Deep Learning(D2L)**: 딥러닝 전반(수학 부담은 챕터 선택)
  - **NLP with Transformers**: Transformer 실전(후반 확장용)

---

## 4) 진행 방식(우리 둘의 역할)
- 나는: 매 주차별 **교재(짧은 설명 + 그림/표 + 체크퀴즈)** + **실습 코드/과제**를 만들어 줌
- 당신: 코드를 실행/수정하면서 막히는 지점 질문(에러 로그 그대로 주면 빠름)

다음 단계: [lessons/01_why_language_model.md](lessons/01_why_language_model.md)부터 레슨/실습을 순서대로 진행하세요.
