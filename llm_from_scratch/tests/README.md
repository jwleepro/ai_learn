# 테스트 안내

이 폴더의 테스트는 두 층으로 나뉩니다.

## 1) 학습자용 빠른 테스트

- 파일 패턴: `test_core_week*.py`
- 목적: 주차별 핵심 흐름이 돌아가는지만 빠르게 확인
- 예:

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week*.py" -v
```

## 2) 전체 단위 테스트

- 파일 패턴: `test_*.py`
- 목적: 내부 유틸리티, 예외 처리, shape, 저장/로드까지 폭넓게 확인
- 예:

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_*.py" -v
```

## 어떻게 쓰면 좋나

- 레슨/과제를 따라가는 중: `test_core_week*.py`
- 코드 수정 후 전체 회귀 확인: `test_*.py`

`test_week0.py`처럼 더 세밀한 파일은 유지보수용 단위 테스트에 가깝고,
`test_core_week0.py`처럼 주차 이름이 붙은 파일은 학습자용 스모크 테스트에 가깝습니다.
