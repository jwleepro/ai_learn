# Week 5 과제: Transformer 블록 감 잡기

목표: Transformer의 “조립 구조”를 실행으로 확인합니다.

---

## 과제 1) forward 데모 실행

```powershell
python llm_from_scratch/code/demo_transformer_forward.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 48 --d_model 64 --heads 4 --layers 2 --top 10
```

질문:

- `--tokens`를 늘리면 무엇이 늘어나나요? (logits의 어떤 차원?)
- `--heads`를 1→4→8로 바꾸면 어떤 제약이 생기나요? (`d_model`과의 관계)
  - 힌트: 보통 `d_model % heads == 0`이어야 하고, `d_head = d_model / heads`입니다.

---

## 과제 2) Attention weights 출력(선택)

Week4의 `demo_self_attention.py`에서:

- `--tokens`를 조금 늘리고
- `--matrix` 옵션으로 전체 weight 행렬을 출력해보세요.

```powershell
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 16 --pos 15 --top 6 --matrix
```
