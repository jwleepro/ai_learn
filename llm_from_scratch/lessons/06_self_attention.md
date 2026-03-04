# 06. Self-Attention: “어디를 볼지”를 학습하는 계산

목표: Transformer의 핵심인 self-attention을 “수식”이 아니라 **계산 흐름**으로 이해하고, 실제 가중치(attention weights)를 눈으로 봅니다.

> 용어/shape(배열 크기)가 헷갈리면 `../GLOSSARY.md`를 먼저 보고 오세요.

---

## 1) 직관

MLP LM은 컨텍스트를 한 덩어리로 펼쳐서 처리합니다.

하지만 어떤 순간에는:

- “바로 직전”이 중요하고
- 어떤 순간에는 “몇 글자 전 단어”가 중요할 수 있어요.

Attention은 말 그대로:

> “지금 위치에서, 과거의 어느 위치를 얼마나 볼지”를 가중치로 만든다.

입니다.

---

## 2) 계산(한 번만)

시퀀스 길이 `T`, 임베딩 차원 `D`가 있을 때:

- 입력 임베딩: `X` (T x D)
- `Q = XWq`, `K = XWk`, `V = XWv`
- 점수: `scores = QK^T / sqrt(Dh)`
- 가중치: `weights = softmax(scores)`
- 출력: `out = weights @ V` (가중합)

여기서 중요한 것은:

- `weights`는 `(T x T)` 행렬이고,
- 각 행(row)은 “해당 위치가 과거 각 위치를 얼마나 보는지”입니다.
  - 예: `weights[i, j]`가 크면, i번째 토큰이 j번째 토큰을 많이 참고합니다.

`scores = QK^T`는 어렵게 보이지만, 코드로 보면 이런 의미입니다:

- `scores[i, j] = dot(Q[i], K[j]) / sqrt(Dh)`

### shape 빠른 표(외우기용)

- `X`: `(T, D)`
- `Wq, Wk, Wv`: `(D, Dh)` (여기서 `Dh`는 head 차원)
- `Q, K, V`: `(T, Dh)`
- `scores`: `(T, T)` (각 위치끼리 “얼마나 비슷한지” 점수)
- `weights`: `(T, T)` (각 행의 합이 1; “어디를 얼마나 볼지”)
- `out`: `(T, Dh)` (V를 weights로 섞은 결과)

> 이 프로젝트의 구현 참고: `llm_from_scratch/code/attention_numpy.py`

---

## 3) Causal mask(미래를 못 보게)

언어모델은 다음 토큰을 맞혀야 하므로,
현재 위치에서 **미래 토큰을 보면 반칙**입니다.

그래서 `j > i`(미래) 위치는 점수를 `-inf`(구현에서는 `-1e9` 같은 큰 음수)로 만들어 softmax 후 0이 되게 합니다.

---

## 4) 실습(데모)

아래는 “학습”이 아니라, **랜덤 가중치**로 attention 가중치의 shape/마스크 동작을 눈으로 보는 데모입니다.

```powershell
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 20 --pos 19 --top 8
```

causal을 끄면(반칙 모드) 무엇이 달라지는지도 비교해보세요:

```powershell
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 20 --pos 19 --top 8 --no_causal
```
