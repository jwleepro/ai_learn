# 다음 단계(Week 2~) 안내

이 문서는 “다음으로 뭘 하면 되지?”를 빠르게 안내합니다.

지금 폴더에는 Week 1~7의 레슨/실습이 이미 준비되어 있습니다.

용어/shape(배열 크기)가 헷갈리면 `GLOSSARY.md`를 먼저 보고 오세요.

---

## 1) Week 2: 신경망 빅램 LM(가장 쉬운 딥러닝 LM)

핵심 아이디어:

- vocab 크기가 `V`일 때, 가중치 행렬 `W`를 `(V x V)`로 둡니다.
- 이전 토큰 ID가 `i`면, `W[i]` 한 줄이 다음 토큰의 “점수(logits)”입니다.
- softmax로 확률로 바꾸고, 정답 next 토큰과 비교해 loss를 계산합니다.
- 경사하강으로 `W`를 업데이트합니다.

좋은 점:

- 구현이 매우 단순합니다(행렬미분을 몰라도 됨).
- “학습한다”는 느낌이 확실히 옵니다.

실습:

- 학습: `python llm_from_scratch/code/train_bigram_nn.py --input llm_from_scratch/data/tiny_corpus_ko.txt`
- 생성: `python llm_from_scratch/code/generate_bigram_nn.py --model llm_from_scratch/models/bigram_nn.npz --length 300 --top_p 0.9`

---

## 2) Week 3: 컨텍스트 늘리기(미니 MLP LM)

빅램은 “직전 1글자”만 보지만, 더 많은 정보를 보려면:

- 입력을 최근 `k`글자(또는 토큰)로 늘리고
- 임베딩을 붙이고
- 작은 MLP로 다음 글자를 예측합니다.

실습:

- 학습: `python llm_from_scratch/code/train_mlp_lm.py --input llm_from_scratch/data/tiny_corpus_ko.txt --context 8`
- 생성: `python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --top_k 40`

---

## 3) Week 4~5: Attention/Transformer 감각 만들기

- 레슨: `lessons/06_self_attention.md`, `lessons/07_transformer_block.md`
- 데모: `python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt`

---

## 4) Week 6: 생성/평가 도구

- 레슨: `lessons/08_sampling_and_eval.md`
- 평가: `python llm_from_scratch/code/evaluate_lm.py ...`

---

## 5) Week 7: BPE 토크나이저

- 레슨: `lessons/09_bpe_tokenizer.md`
- 학습: `python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 200`

---

## 6) LLM 트랙(Transformer)로 넘어갈 때

Transformer를 “학습까지” 하려면 자동미분이 있는 프레임워크가 현실적입니다.

- 권장: **Python 3.12/3.13 + PyTorch**
- 이 폴더의 현재 환경(Python 3.14)에서는 PyTorch가 미지원/미설치일 수 있습니다.

준비가 되면:

1. 별도 venv에서 Python 3.12/3.13 설치
2. PyTorch 설치
3. (우리가 다음 단계에서 제공할) `MiniGPT` 실습을 진행

---

## 7) 지금 할 일(당장)

1. `exercises/WEEK1.md`의 과제 1~4를 해보기
2. `WEEK2.md` → `WEEK3.md` 순서로 학습/생성
3. `WEEK6.md`로 샘플링/평가 감 잡기
4. `WEEK7.md`로 BPE 튜닝

---

## 8) (선택) 파인튜닝 확장(용어/운영 감각)

파인튜닝 글을 읽을 때 용어가 너무 많아 막히면, 아래 2개 레슨부터 보세요.

- `lessons/11_finetuning_essentials.md`
- `lessons/12_lora_qlora_and_ops.md`

간단 과제(데이터/평가 준비):

- `exercises/WEEK8.md`
