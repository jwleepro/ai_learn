# Week 6 과제: 샘플링/평가로 모델 비교

목표: “모델 개선”과 “생성 품질”을 분리해서 다루는 감각을 잡습니다.

---

## 과제 1) 샘플링 옵션 비교(빅램/신경망/MLP)

각 모델에서 아래 조합을 비교해보세요.

- `--temperature 0.8`
- `--temperature 1.0`
- `--temperature 1.2`
- `--top_k 40` (또는 vocab이 작으면 10)
- `--top_p 0.9`

예시(카운트 빅램):

```powershell
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 0 --temperature 1.0 --top_p 0.9
```

---

## 과제 2) perplexity로 비교

학습한 3개 모델을 같은 eval 텍스트로 평가해보세요.

```powershell
python llm_from_scratch/code/evaluate_lm.py counts_bigram --train llm_from_scratch/data/tiny_corpus_ko.txt --eval llm_from_scratch/data/tiny_corpus_ko.txt --smoothing 1
python llm_from_scratch/code/evaluate_lm.py bigram_nn --model llm_from_scratch/models/bigram_nn.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
python llm_from_scratch/code/evaluate_lm.py mlp_lm --model llm_from_scratch/models/mlp_lm.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
```

질문:

1. perplexity가 가장 낮은 모델이 항상 “가장 재미있는 생성”인가요?
2. 생성 품질을 더 잘 평가하려면 무엇이 필요할까요? (힌트: task/벤치마크/사람 평가)

