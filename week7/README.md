# Week 7 과제 (Core / numpy)

> 관련 레슨: [09_bpe_tokenizer](lessons/09_bpe_tokenizer.md)

목표: 현대 LLM에서 필수적으로 사용하는 **BPE(Byte Pair Encoding)** 토크나이저의 학습 및 인코딩 원리를 이해합니다.

---

## 과제 1) BPE 토크나이저 학습

아래 명령을 실행하여 텍스트 데이터로부터 BPE 병합 규칙을 추출하고 토크나이저를 학습시키세요.

> 소스: [`week7_complete.py`](code/week7_complete.py)
```powershell
python code/week7_complete.py --train --vocab_size 500 --data data/tiny_corpus_ko.txt
```

질문:
1. 병합이 진행됨에 따라 토큰의 모양(글자 조합)이 어떻게 변하나요?
2. `vocab_size`를 256(기본 바이트)에서 500, 1000으로 늘렸을 때 어떤 차이가 있나요?

---

## 과제 2) BPE 인코딩/디코딩 데모

학습된 토크나이저를 사용하여 임의의 문장을 인코딩하고 다시 디코딩해보세요.

> 소스: [`week7_complete.py`](code/week7_complete.py)
```powershell
python code/week7_complete.py --demo
```

질문:
1. 하나의 토큰이 여러 글자를 포함하고 있나요?
2. 문자 단위 토크나이저보다 텍스트를 표현하는 데 필요한 토큰의 수가 줄어들었나요?

---

## 자기 점검(자동)

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week7_bpe.py" -v
```
