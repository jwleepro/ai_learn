# Week 6 과제 (Core / numpy)

> 관련 레슨: [08_sampling_and_eval](lessons/08_sampling_and_eval.md)

목표: 학습된 언어모델의 성능을 **Perplexity**라는 지표를 통해 정량적으로 평가해봅니다.

---

## 과제 1) 언어모델 성능 평가 (Loss & PPL)

아래 명령을 실행하여 이전에 학습한 모델들의 성능을 비교해보세요.

> 소스: [`week6_complete.py`](code/week6_complete.py)

1. **Count-based Bigram**:
```powershell
python code/week6_complete.py --counts --data data/tiny_corpus_ko.txt
```

2. **Neural Bigram** (학습된 모델이 필요합니다):
```powershell
python code/week6_complete.py --bigram_nn --model_path week2/code/bigram_nn_model.npz --data data/tiny_corpus_ko.txt
```

3. **MLP LM** (학습된 모델이 필요합니다):
```powershell
python code/week6_complete.py --mlp_lm --model_path week3/code/mlp_model.npz --data data/tiny_corpus_ko.txt
```

질문:
1. 어떤 모델이 가장 낮은 Perplexity(더 좋은 성능)를 기록했나요?
2. Perplexity가 낮을수록 생성된 문장이 더 "말이 된다"고 느껴지나요?

---

## 과제 2) 평가 지표의 이해

`week6_complete.py`의 `calculate_metrics` 함수를 참고하여, Cross Entropy Loss가 어떻게 Perplexity로 변환되는지 수식을 확인하고 그 의미를 설명해보세요.

---

## 자기 점검(자동)

(Week 6은 평가 위주이므로 별도의 단위 테스트는 생략합니다.)
