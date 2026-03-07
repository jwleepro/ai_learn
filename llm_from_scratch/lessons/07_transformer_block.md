# 07. Transformer 블록 조립하기(구성요소 이해)

목표: Transformer가 사실 “몇 가지 레고 블록”의 반복임을 이해합니다.

이번 주는 (현재 폴더의 numpy 환경 기준) **전방향(forward) 계산 흐름**을 확인하는 데 초점을 둡니다.  
학습(역전파 포함)은 PyTorch 트랙에서 다루는 것이 현실적입니다.

> 용어/shape(배열 크기)가 헷갈리면 [GLOSSARY_CORE](../GLOSSARY_CORE.md)를 먼저 보고 오세요.

---

## 1) Transformer 블록(한 레이어) 구성

대부분의 GPT류 모델은 레이어마다 이런 구조를 반복합니다:

1. LayerNorm
2. Multi-Head Self-Attention (+ Residual)
3. LayerNorm
4. FeedForward MLP (+ Residual)

그리고 마지막에:

- Final LayerNorm
- Linear projection → vocab logits

### 기호 없이 먼저 보면

한 블록은 아래 흐름으로 보면 됩니다.

1. 입력 벡터를 잠깐 정리한다. (`LayerNorm`)
2. 지금 필요한 과거 정보만 골라 섞는다. (`Attention`)
3. 원래 입력을 다시 더한다. (`Residual`)
4. 작은 MLP로 한 번 더 계산한다.
5. 또 원래 값을 더한다. (`Residual`)

### shape 빠른 표(외우기용)

아래는 “대충 이런 크기구나”만 잡으면 됩니다.

- 입력 token ids: `(T,)` (토큰 T개)
- 토큰 임베딩/은닉 상태 `x`: `(T, d_model)`
- attention weights(헤드별): `(n_heads, T, T)`
- 최종 logits: `(T, V)` (각 위치마다 vocab 크기만큼 점수)

중요 제약:

- 보통 `d_model % n_heads == 0` 이어야 하고,
- `d_head = d_model / n_heads`로 나눠서 각 헤드가 처리합니다.

---

## 2) Residual(스킵 연결)이 중요한 이유

`x = x + f(x)` 형태는:

- 깊은 네트워크에서도 학습이 안정적
- “수정해야 할 것만” 조금씩 바꾸기 쉬움

으로 이해하면 충분합니다.

---

## 3) LayerNorm은 무엇을 하나?

토큰 벡터 한 개(길이 D)를:

- 평균 0, 분산 1 근처로 정규화
- 학습 가능한 스케일/시프트로 다시 조정

해서 학습을 안정화합니다.

논문에서는 이 스케일/시프트를 `(γ, β)`라고 쓰고,
코드에서는 보통 `g, b` 또는 `scale, shift`처럼 더 평범한 이름으로 둡니다.

---

## 4) 실습: 랜덤 가중치 Transformer forward

학습은 아니지만, “블록이 어떻게 붙는지”를 실행으로 확인합니다.

```powershell
python llm_from_scratch/code/demo_transformer_forward.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 64 --d_model 64 --heads 4 --layers 2
```

출력은 랜덤 예측이지만, 다음을 확인하면 목표 달성:

- 입력 토큰 수(T)만큼 logits이 나온다
- 마지막 위치 logits으로 “다음 토큰 분포”를 만든다

> 구현 참고: `llm_from_scratch/code/transformer_numpy.py`

---

---

과제: [WEEK5](../exercises/WEEK5.md)

[← 이전: Self-Attention](06_self_attention.md) | [목차](INDEX.md) | [다음: 샘플링 & 평가 →](08_sampling_and_eval.md)
