# Week 4 과제 (Core / numpy)

> 관련 레슨: [06_self_attention](lessons/06_self_attention.md)

목표: 트랜스포머의 핵심인 **Self-Attention**의 동작 원리를 NumPy 구현을 통해 이해합니다.

---

## 과제 1) Self-Attention 가중치 분석

아래 명령을 실행하여 특정 위치의 토큰이 다른 토큰들을 어떻게 "주목(attend)"하는지 확인하세요.

> 소스: [`week4_complete.py`](code/week4_complete.py)
```powershell
python code/week4_complete.py --input data/tiny_corpus_ko.txt --tokens 20 --pos 19
```

질문:
1. `causal=True`일 때와 아닐 때(`--no_causal` 추가), 어텐션 가중치의 분포가 어떻게 달라지나요?
2. 특정 조사나 어미 뒤에 올 때 어텐션이 쏠리는 위치가 있나요?

---

## 과제 2) Causal Masking 이해

`week4_complete.py` 코드 내의 `causal_mask` 함수가 어떻게 행렬의 윗부분을 `-1e9`로 채우는지 확인하고, 이것이 Softmax를 거친 후 어떤 결과가 되는지 설명해보세요.

---

## 자기 점검(자동)

(Week 4는 학습 코드가 포함되어 있지 않으므로, 별도의 단위 테스트는 생략하거나 구현 검증 위주로 진행합니다.)
