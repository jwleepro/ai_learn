# 08. 생성(샘플링) & 평가(perplexity)

목표:

- 같은 모델이라도 “어떻게 뽑느냐”에 따라 출력 품질이 크게 달라짐을 이해합니다.
- loss/perplexity로 모델을 비교하는 최소 평가를 합니다.

---

## 1) temperature

- `temperature < 1`: 더 보수적(반복/안정)
- `temperature = 1`: 기본
- `temperature > 1`: 더 다양(랜덤/깨짐 증가 가능)

---

## 2) top-k

확률이 큰 상위 k개 후보만 남기고 나머지는 버립니다.

- 장점: 말도 안 되는 토큰 선택을 줄임
- 단점: 다양성이 줄 수 있음

---

## 3) top-p (nucleus sampling)

확률을 큰 순서로 더해서 누적합이 `p`가 될 때까지의 후보만 남깁니다.

- 장점: 상황에 따라 후보 개수가 자동으로 조절됨
- 실전에서 자주 씀

> 구현 참고: `llm_from_scratch/code/sampling.py`

---

## 4) perplexity (아주 간단히)

모델이 “정답 다음 토큰”에 얼마나 확신을 주는지의 지표입니다.

- loss가 낮을수록 좋고
- perplexity도 낮을수록 좋습니다.

> 구현 참고: `llm_from_scratch/code/evaluate_lm.py`

---

## 5) 실습 예시

### 생성 옵션 비교(MLP LM)

```powershell
python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --seed 0 --temperature 1.0 --top_k 40
python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --seed 0 --temperature 1.0 --top_p 0.9
```

### 평가(같은 eval 텍스트로 비교)

```powershell
python llm_from_scratch/code/evaluate_lm.py counts_bigram --train llm_from_scratch/data/tiny_corpus_ko.txt --eval llm_from_scratch/data/tiny_corpus_ko.txt --smoothing 1
python llm_from_scratch/code/evaluate_lm.py bigram_nn --model llm_from_scratch/models/bigram_nn.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
python llm_from_scratch/code/evaluate_lm.py mlp_lm --model llm_from_scratch/models/mlp_lm.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
```

---

과제: [WEEK6](../exercises/WEEK6.md)

[← 이전: Transformer 블록](07_transformer_block.md) | [목차](INDEX.md) | [다음: BPE 토크나이저 →](09_bpe_tokenizer.md)
