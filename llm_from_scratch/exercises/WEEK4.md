# Week 4 과제: Attention 감각 만들기

> 관련 레슨: [06_self_attention — Self-Attention](../lessons/06_self_attention.md)

목표: “attention 가중치가 무엇을 의미하는지”를 직접 출력해보고 이해합니다.

---

## 과제 1) causal vs non-causal 비교

아래 두 명령을 실행해서 비교하세요.

```powershell
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 24 --pos 23 --top 8
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 24 --pos 23 --top 8 --no_causal
```

질문:

- non-causal에서 “미래”로 가는 weight가 생기나요?
- 언어모델 학습에서 non-causal이 왜 문제가 되나요?

---

## 과제 2) 위치를 바꿔가며 관찰

`--pos`를 5, 10, 15, 23처럼 바꾸면서:

- 어떤 위치는 “바로 이전”을 많이 보고
- 어떤 위치는 “조금 더 과거”를 보는지

관찰해보세요.

