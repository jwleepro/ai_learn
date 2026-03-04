# 00. 실습 환경 준비

목표: “코드를 실행할 수 있는 최소 환경”을 만들고, 앞으로의 실습이 막히지 않게 합니다.

## 1) 지금 필요한 것

- Python (권장: 3.12~3.13 / Core 트랙은 3.14도 가능)
- `numpy`

> 이 작업 폴더 기준으로는 `Python 3.14`와 `numpy`가 이미 설치되어 있습니다.

## 2) (권장) 가상환경 만들기

Windows PowerShell 예시:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r llm_from_scratch/requirements-core.txt
```

> (선택) PyTorch 트랙은 별도 venv에서 `llm_from_scratch/requirements-torch.txt`를 설치합니다.  
> Python 3.14에서는 PyTorch가 미지원일 수 있으니 `lessons/10_minigpt_pytorch.md`의 권장 버전을 참고하세요.

## 3) 체크(정상 동작 확인)

아래가 실행되면 OK:

```powershell
python -c "import numpy as np; print('numpy', np.__version__)"
```

## 4) 다음 레슨

`01_why_language_model.md`에서 “언어모델이 실제로 무엇을 학습하는가”를 코드 관점으로 잡습니다.
