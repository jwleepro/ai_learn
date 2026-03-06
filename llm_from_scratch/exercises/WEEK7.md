# Week 7 과제: BPE 토크나이저 튜닝

> 관련 레슨: [09_bpe_tokenizer — BPE 토크나이저 직접 만들기](../lessons/09_bpe_tokenizer.md)

목표: merge 수를 바꿔가며 토큰이 어떻게 바뀌는지 관찰합니다.

---

## 과제 1) merge 수 비교

아래를 각각 실행하세요:

```powershell
python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 50  --out llm_from_scratch/models/bpe_50.json
python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 200 --out llm_from_scratch/models/bpe_200.json
python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 800 --out llm_from_scratch/models/bpe_800.json
```

각 토크나이저로 같은 텍스트를 인코딩해 비교:

```powershell
python llm_from_scratch/code/demo_bpe.py --tokenizer llm_from_scratch/models/bpe_50.json  --text_file llm_from_scratch/data/tiny_corpus_ko.txt --max_tokens 40
python llm_from_scratch/code/demo_bpe.py --tokenizer llm_from_scratch/models/bpe_200.json --text_file llm_from_scratch/data/tiny_corpus_ko.txt --max_tokens 40
python llm_from_scratch/code/demo_bpe.py --tokenizer llm_from_scratch/models/bpe_800.json --text_file llm_from_scratch/data/tiny_corpus_ko.txt --max_tokens 40
```

질문:

1. merge가 늘수록 “토큰 길이”와 “토큰 개수”는 어떻게 바뀌나요?
2. 너무 많은 merge는 어떤 문제를 만들까요? (힌트: 과적합/희귀 토큰)

