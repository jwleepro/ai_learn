# 05. 컨텍스트를 늘리는 첫 모델: MLP 언어모델

목표: “직전 1글자만 보는 빅램”에서 벗어나, 최근 `k`개 글자를 보고 다음 글자를 예측하는 모델을 직접 학습합니다.

> 용어/shape(배열 크기)가 헷갈리면 [GLOSSARY](../GLOSSARY.md)를 먼저 보고 오세요.

---

## 1) 왜 컨텍스트가 필요할까?

빅램은 `prev 한 글자`만 봅니다. 그래서 이런 문제가 생깁니다.

- 같은 글자 `다`라도, 앞의 문맥에 따라 다음이 완전히 달라짐
- 문장 구조(조사/어미/띄어쓰기)처럼 “몇 글자 전 정보”가 중요함

따라서 입력을 이렇게 늘립니다:

- 입력: 최근 `k`개 토큰 `[x_{t-k}, ..., x_{t-1}]`
- 정답: 다음 토큰 `x_t`

---

## 2) 임베딩(Embedding): 토큰을 벡터로

토큰 ID는 그냥 정수입니다. 모델은 벡터를 좋아하니,

- 임베딩 테이블 `E` (크기: `V x D`)
- 토큰 ID `i`는 벡터 `E[i]`로 변환

여기서:

- `V`: vocab 크기(등장 글자 수)
- `D`: 임베딩 차원(우리가 정함)

---

## 3) MLP 모델 구조(아주 단순)

컨텍스트 길이가 `k`면:

1. 각 토큰을 임베딩으로 바꾼다 → `(k, D)`
2. 이를 한 줄로 펼친다(concat) → `(k*D,)`
3. 은닉층(hidden) 한 번 통과 → `(H,)`
4. 출력층으로 vocab 크기만큼 점수(logits) → `(V,)`
5. softmax로 확률화 → `P(next | context)`

이게 끝입니다.

### shape 빠른 표(외우기용)

여기서 `B`는 batch 크기(한 번에 처리하는 샘플 수)입니다.

- 입력 `X`: `(B, k)`  (토큰 id k개가 한 샘플)
- 임베딩 `emb = E[X]`: `(B, k, D)`
- 펼치기 `h_in`: `(B, k*D)`
- 은닉 `h`: `(B, H)`
- 출력 logits: `(B, V)`

> 구현 참고: `llm_from_scratch/code/mlp_lm.py`  
> (미분 유도는 생략하고, “shape/흐름” 위주로 따라가도 됩니다.)

---

## 4) 학습 데이터 만들기(중요)

텍스트 토큰 ID가:

`[x0, x1, x2, x3, x4, ...]`

컨텍스트 길이 `k=3`이면 샘플은:

- 입력: `[x0, x1, x2]` → 정답: `x3`
- 입력: `[x1, x2, x3]` → 정답: `x4`
- ...

---

## 5) 실습(우리가 제공한 코드)

학습:

```powershell
python llm_from_scratch/code/train_mlp_lm.py --input llm_from_scratch/data/tiny_corpus_ko.txt --context 8 --embed 32 --hidden 128 --epochs 60 --lr 0.2
```

생성:

```powershell
python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --seed 0 --temperature 1.0
```

---

## 6) 체크 퀴즈

1. `context`를 2→8→16으로 키우면, 어떤 점이 좋아지고 어떤 점이 나빠질까요?
2. `embed`와 `hidden`을 키우면 어떤 일이 생길까요?
3. `epochs`를 늘리면 항상 좋아질까요? (train vs val 관찰)

---

과제: [WEEK3](../exercises/WEEK3.md)

[← 이전: 신경망 빅램 LM](04_neural_bigram.md) | [목차](INDEX.md) | [다음: Self-Attention →](06_self_attention.md)
