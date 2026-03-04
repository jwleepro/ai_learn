# 09. BPE 토크나이저(서브워드) 직접 만들기

목표:

- 왜 LLM은 “단어”도 “글자”도 아닌 **서브워드**를 쓰는지 이해합니다.
- BPE(Byte Pair Encoding)의 핵심 아이디어를 직접 구현으로 익힙니다.

> 참고: 이 프로젝트의 BPE는 “학습용 단순 버전”입니다.  
> 공백 기준으로 단어를 나누기 때문에, `decode()`가 원문의 공백/줄바꿈을 100% 복원하지는 않습니다(의도된 단순화).

---

## 1) 왜 서브워드인가?

단어 토큰화는 어휘가 너무 커지고(희귀 단어),
글자 토큰화는 시퀀스가 너무 길어집니다.

서브워드는 그 중간:

- 자주 나오는 조각은 하나의 토큰으로 합치고
- 드문 단어는 여러 조각으로 쪼개서 처리

합니다.

---

## 2) BPE의 핵심 아이디어

1. 처음에는 “아주 작은 단위”(보통 문자/바이트)로 시작
2. 말뭉치에서 **가장 자주 붙어 나오는 인접 쌍(pair)**을 찾음
3. 그 pair를 하나의 새 토큰으로 **merge**
4. 이 과정을 여러 번 반복

결과:

- 자주 등장하는 조각이 점점 길어진 토큰으로 합쳐집니다.

---

## 3) 실습 코드

학습(merge 200번):

```powershell
python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 200
```

데모(encode/decode):

```powershell
python llm_from_scratch/code/demo_bpe.py --tokenizer llm_from_scratch/models/bpe_tokenizer.json --text_file llm_from_scratch/data/tiny_corpus_ko.txt --max_tokens 40
```

> 구현 참고: `llm_from_scratch/code/bpe_tokenizer.py`

---

## 4) 체크 퀴즈

1. merge 횟수를 50→200→1000으로 늘리면 어떤 변화가 생길까요?
2. vocab이 너무 커지면 모델/학습에는 어떤 부담이 생길까요?
