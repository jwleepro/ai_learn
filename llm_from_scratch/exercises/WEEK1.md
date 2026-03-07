# Week 1 과제 (Core / numpy)

> 관련 레슨: [01_why_language_model](../lessons/01_why_language_model.md), [02_tokenization_char](../lessons/02_tokenization_char.md), [03_bigram_counts](../lessons/03_bigram_counts.md)

목표: “다음 글자 예측”이 실제로 돌아간다는 감각을 잡고, 결과를 스스로 분석해봅니다.

---

## 과제 1) 토크나이저 왕복(encode/decode) 확인

1. `llm_from_scratch/data/tiny_corpus_ko.txt`를 읽는다.
2. 토크나이저를 만든다.
3. `encode -> decode`를 거쳤을 때 원문과 동일한지 확인한다.

체크: “같다/다르다”와, 다르면 첫 불일치 위치를 출력해보기.

---

## 과제 2) 스무딩 효과 보기

아래 두 명령을 실행하고, 생성 텍스트의 차이를 관찰하세요.

```powershell
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 0 --smoothing 0
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 0 --smoothing 1
```

질문:

- 스무딩을 켜면 어떤 전이가 “가능해졌다”고 느껴지나요?
- 대신 어떤 단점이 생기나요?

---

## 과제 3) 온도(temperature) 효과 보기

아래를 각각 실행하고, 결과를 비교하세요.

```powershell
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 1 --temperature 0.7
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 1 --temperature 1.0
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 1 --temperature 1.3
```

질문:

- `0.7` / `1.0` / `1.3`의 느낌을 한 줄로 정리하면?
- “너무 랜덤”과 “너무 반복” 사이에서 어떤 값이 좋았나요?

---

## 과제 4) 내 코퍼스로 바꿔보기

1. `llm_from_scratch/data/my_corpus.txt`를 새로 만들고, 본인이 원하는 문장/대화/메모를 200~2000자 넣어보세요.
2. 같은 명령으로 생성해보고, `tiny_corpus_ko.txt`와 결과가 어떻게 다른지 비교해보세요.

---

## (선택) 과제 5) 전이 확률을 눈으로 확인하기

특정 글자 뒤에 어떤 글자가 잘 나오는지 확인해보세요:

```powershell
python llm_from_scratch/code/inspect_bigrams.py --input llm_from_scratch/data/tiny_corpus_ko.txt --char_u 0xB2E4 --top 10
```

질문:

- 내가 예상한 전이가 실제로 top에 있나요?
- “띄어쓰기”나 “줄바꿈” 같은 글자는 어떤 역할을 하나요?

---

## 자기 점검(자동)

아래가 통과하면 Week 1 핵심 흐름은 OK입니다:

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week1.py" -v
```

전체 단위 테스트가 필요하면:

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_*.py" -v
```
