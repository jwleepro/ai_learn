# Week 5 과제: Transformer 블록 감 잡기

> 관련 레슨: [07_transformer_block — Transformer 블록 조립하기](lessons/07_transformer_block.md)

목표: Transformer의 “조립 구조”를 실행으로 확인합니다.

---

## 과제 1) forward 데모 실행

> 소스: [`demo_transformer_forward.py`](code/demo_transformer_forward.py)
```powershell
python code/demo_transformer_forward.py --input data/tiny_corpus_ko.txt --tokens 48 --d_model 64 --heads 4 --layers 2 --top 10
```

질문:

- `--tokens`를 늘리면 무엇이 늘어나나요? (logits의 어떤 차원?)
- `--heads`를 1→4→8로 바꾸면 어떤 제약이 생기나요? (`d_model`과의 관계)
  - 힌트: 보통 `d_model % heads == 0`이어야 하고, `d_head = d_model / heads`입니다.

---

## 과제 2) Attention weights 출력(선택)

> 소스: [`demo_self_attention.py`](code/demo_self_attention.py)

아래 명령으로 attention weights를 시각화해보세요:

```powershell
python code/demo_self_attention.py --input data/tiny_corpus_ko.txt --tokens 24 --pos 23 --top 8
```

- `--tokens`를 조금 늘리고
- `--matrix` 옵션으로 전체 weight 행렬을 출력해보세요.

> 소스: [`demo_self_attention.py`](code/demo_self_attention.py)
```powershell
python code/demo_self_attention.py --input data/tiny_corpus_ko.txt --tokens 16 --pos 15 --top 6 --matrix
```
