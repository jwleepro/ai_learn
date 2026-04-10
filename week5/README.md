# Week 5 과제 (Core / numpy)

> 관련 레슨: [07_transformer_block](lessons/07_transformer_block.md)

목표: **Multi-Head Attention**과 **Transformer Block**의 전체적인 Forward Pass 과정을 이해합니다.

---

## 과제 1) Transformer Forward Pass 실행

아래 명령을 실행하여 트랜스포머 모델이 입력 토큰에 대해 다음 토큰 확률을 어떻게 계산하는지 확인하세요.

> 소스: [`week5_complete.py`](code/week5_complete.py)
```powershell
python code/week5_complete.py --input data/tiny_corpus_ko.txt --tokens 32 --d_model 64 --n_heads 4
```

질문:
1. 헤드 수(`--n_heads`)를 바꿨을 때 모델의 동작(출력 형태 등)에 변화가 있나요?
2. `d_model`이 각 컴포넌트(MHA, FFN)에서 어떻게 사용되는지 코드를 통해 확인해보세요.

---

## 과제 2) 블록 구조 파악

`week5_complete.py`의 `main` 함수 내에서 Residual Connection과 Layer Normalization이 적용되는 순서를 확인하고, 왜 이런 구조를 사용하는지 생각해보세요.

---

## 자기 점검(자동)

(Week 5는 학습 코드가 포함되어 있지 않으므로, 별도의 단위 테스트는 생략하거나 구현 검증 위주로 진행합니다.)
