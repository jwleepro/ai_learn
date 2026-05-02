# AI 학습 — Java/C# 개발자를 위한 LLM 입문

핸드폰으로도 읽기 쉽게, **한 주 = 한 파일 = 한 개념**.
LLM(거대 언어모델)이 어떻게 동작하는지 numpy 만으로 바닥부터 만들어 본다.

---

## 누구를 위한 자료인가

- Java / C# / JavaScript 는 익숙하다
- 행렬·벡터는 들어는 봤지만 제대로 써본 적은 없다
- "LLM 이 도대체 안에서 뭐 하는 거지?" 가 가장 궁금하다

수학은 중학교 수준이면 충분하다. 모든 수식은 코드로 풀어쓴다.

---

## 10주 진행표

| 주 | 파일 | 한 줄 요약 |
|---|---|---|
| 1 | `week1.py` | AI 큰 그림 + numpy 5분 입문 (행렬곱이 곧 신경망) |
| 2 | `week2.py` | 퍼셉트론과 경사하강법 — "학습" 이 뭔지 한 변수 회귀로 체감 |
| 3 | `week3.py` | 빅램(2-gram) 카운트 언어모델 — 확률표만으로 텍스트 생성 |
| 4 | `week4.py` | 임베딩 + 소프트맥스 — 단어를 벡터로, 벡터를 확률로 |
| 5 | `week5.py` | 크로스엔트로피 손실과 역전파 — 가중치를 어디로 움직이는가 |
| 6 | `week6.py` | MLP 언어모델 — 컨텍스트 확장 + 학습/검증 분리 + 퍼플렉서티 |
| 7 | `week7.py` | Self-Attention — Q/K/V 와 causal mask |
| 8 | `week8.py` | Transformer 블록 — Multi-Head + LayerNorm + 잔차 + FFN |
| 9 | `week9.py` | MiniGPT — 블록 N개 쌓아 numpy 만으로 학습 + 샘플링 |
| 10 | `week10.py` | 다음 단계 — PyTorch 입문 + BPE / SFT 데이터 미리보기 |

---

## 실행 방법

```bash
python week1.py
python week2.py
# ...
python week10.py
```

- `week1.py` ~ `week9.py` : `numpy` 만 있으면 된다.
- `week10.py` : `torch` 설치 필요 (없으면 numpy 부분만 실행됨).

데이터·설명·코드 모두 한 파일에 들어 있다. 외부 리소스 없음.

---

## 코드 스타일 약속

Java/C#/JS 개발자가 처음 봐도 흐름이 읽히도록 다음 규칙을 따른다.

- 클래스는 평범한 `__init__` 만 쓰는 클래스. 데코레이터 / dataclass 안 씀.
- 컴프리헨션 안 쓰고 명시적 `for` 루프.
- numpy 의 fancy indexing(`np.add.at` 등) 안 쓰고 풀어쓴 루프.
- Python 특유의 트릭(walrus `:=`, `*` 키워드 only 등) 안 씀.
- `@` (행렬곱) 와 broadcasting 처럼 **AI 본질** 인 것은 유지하되 한 줄 주석으로 의미 설명.
- Java/C# 개발자가 처음 보면 헷갈릴 만한 곳에 **"// Java 비유:"** 형태 주석.

> 목표는 Python 잘 쓰기가 아니라 **LLM 동작 원리 이해** 다.
