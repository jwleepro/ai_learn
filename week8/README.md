# Week 8 과제 (SFT & Ops)

> 관련 레슨: [11_finetuning_essentials](lessons/11_finetuning_essentials.md), [12_lora_qlora_and_ops](lessons/12_lora_qlora_and_ops.md)

목표: 파인튜닝(SFT)을 위한 데이터셋의 형식을 이해하고, 실제 학습에 사용하기 전에 유효성을 검증하는 방법을 익힙니다.

---

## 과제 1) SFT JSONL 데이터 검증

제공된 또는 직접 만든 SFT용 JSONL 데이터셋이 올바른 형식인지 검증하세요.

> 소스: [`week8_complete.py`](code/week8_complete.py)
```powershell
python code/week8_complete.py --input data/sft_toy.jsonl
```

질문:
1. JSONL 파일의 각 줄은 어떤 구조로 되어 있어야 하나요? (필수 키 확인)
2. 데이터에 빈 문자열이 포함되어 있을 때 검증기가 어떻게 반응하나요?

---

## 과제 2) (선택) JSON 출력 검증

모델이 반드시 JSON 형식으로 답변해야 하는 태스크(구조화된 추출)의 경우, 아래 옵션을 사용하여 `output` 필드 내의 JSON 유효성을 추가로 검사해보세요.

```powershell
python code/week8_complete.py --input data/sft_toy.jsonl --expect_json
```

---

## 자기 점검(자동)

(Week 8은 데이터 검증 도구 위주이므로 별도의 단위 테스트는 생략합니다.)
