# LLM 바닥부터 만들기 (실습형, 한국어)

이 폴더는 “LLM을 바닥부터 직접 만들어보는” 학습 프로젝트입니다.  
수학은 최소로, 코드는 최대한으로 진행합니다.

처음에는 [GLOSSARY_CORE.md](GLOSSARY_CORE.md)만 보면 충분합니다.  
파인튜닝/LoRA/운영 용어까지 필요해질 때 [GLOSSARY.md](GLOSSARY.md)를 보세요.

## 시작 순서

1. [CURRICULUM.md](CURRICULUM.md) : 전체 로드맵(주차별 목표/산출물)
2. [GLOSSARY_CORE.md](GLOSSARY_CORE.md) : Core 트랙용 용어/shape 빠른 정리
3. [lessons/00_setup.md](lessons/00_setup.md) : 실습 환경 준비
4. [lessons/00_dl_basics.md](lessons/00_dl_basics.md) : (Week 0) 딥러닝 기초를 코드로
5. [lessons/01_why_language_model.md](lessons/01_why_language_model.md) → [02_tokenization_char.md](lessons/02_tokenization_char.md) → [03_bigram_counts.md](lessons/03_bigram_counts.md) → [04_neural_bigram.md](lessons/04_neural_bigram.md)
6. [lessons/05_mlp_context_lm.md](lessons/05_mlp_context_lm.md) → [06_self_attention.md](lessons/06_self_attention.md) → [07_transformer_block.md](lessons/07_transformer_block.md)
7. [lessons/08_sampling_and_eval.md](lessons/08_sampling_and_eval.md) → [09_bpe_tokenizer.md](lessons/09_bpe_tokenizer.md)
8. (선택) [lessons/10_minigpt_pytorch.md](lessons/10_minigpt_pytorch.md) : PyTorch 문법 난도를 포함한 MiniGPT 트랙
9. (선택) [lessons/11_finetuning_essentials.md](lessons/11_finetuning_essentials.md) → [12_lora_qlora_and_ops.md](lessons/12_lora_qlora_and_ops.md) : 파인튜닝/운영 감각
10. (선택) [GLOSSARY.md](GLOSSARY.md) : 전체 용어집(확장 용어 포함)

과제는 [exercises/INDEX.md](exercises/INDEX.md)에서 주차별로 진행합니다.

## Quickstart (Week 0)

```powershell
python llm_from_scratch/code/demo_week0_dl_basics.py
```

## Quickstart (Week 1)

빅램 카운트 모델로 텍스트 생성:

```powershell
python llm_from_scratch/code/generate_bigram.py --input llm_from_scratch/data/tiny_corpus_ko.txt --length 300 --seed 0
```

특정 글자 뒤 전이확률 보기(예: "다" = `U+B2E4`):

```powershell
python llm_from_scratch/code/inspect_bigrams.py --input llm_from_scratch/data/tiny_corpus_ko.txt --char_u 0xB2E4 --top 10
```

빠른 자기점검(학습자용 주차 테스트):

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week*.py" -v
```

전체 단위 테스트:

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_*.py" -v
```

테스트 파일 구분은 [tests/README.md](tests/README.md)를 참고하세요.

## Week 2~3 (학습/생성)

신경망 빅램 학습/생성:

```powershell
python llm_from_scratch/code/train_bigram_nn.py --input llm_from_scratch/data/tiny_corpus_ko.txt
python llm_from_scratch/code/generate_bigram_nn.py --model llm_from_scratch/models/bigram_nn.npz --length 300 --seed 0 --temperature 1.0 --top_p 0.9
```

MLP LM 학습/생성:

```powershell
python llm_from_scratch/code/train_mlp_lm.py --input llm_from_scratch/data/tiny_corpus_ko.txt --context 8 --embed 32 --hidden 128
python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --seed 0 --temperature 1.0 --top_k 40
```

평가(loss/ppl):

```powershell
python llm_from_scratch/code/evaluate_lm.py counts_bigram --train llm_from_scratch/data/tiny_corpus_ko.txt --eval llm_from_scratch/data/tiny_corpus_ko.txt --smoothing 1
python llm_from_scratch/code/evaluate_lm.py bigram_nn --model llm_from_scratch/models/bigram_nn.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
python llm_from_scratch/code/evaluate_lm.py mlp_lm --model llm_from_scratch/models/mlp_lm.npz --eval llm_from_scratch/data/tiny_corpus_ko.txt
```

## Week 4~7 (데모/도구)

self-attention 데모:

```powershell
python llm_from_scratch/code/demo_self_attention.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 24 --pos 23 --top 8
```

Transformer forward 데모(랜덤 가중치, shape 확인용):

```powershell
python llm_from_scratch/code/demo_transformer_forward.py --input llm_from_scratch/data/tiny_corpus_ko.txt --tokens 64 --d_model 64 --heads 4 --layers 2
```

BPE 토크나이저 학습/데모:

```powershell
python llm_from_scratch/code/train_bpe_tokenizer.py --input llm_from_scratch/data/tiny_corpus_ko.txt --merges 200
python llm_from_scratch/code/demo_bpe.py --tokenizer llm_from_scratch/models/bpe_tokenizer.json --text_file llm_from_scratch/data/tiny_corpus_ko.txt --max_tokens 40
```

## (선택) MiniGPT 학습(PyTorch)

[lessons/10_minigpt_pytorch.md](lessons/10_minigpt_pytorch.md) 참고.

## 우리가 만들 것(요약)

- (Core/numpy) 빅램 카운트 LM → 신경망 빅램 LM → 미니 MLP LM
- (선택/PyTorch) Self-Attention → MiniGPT(작은 Transformer LM)

## 진행 팁

- 막히면 **에러 로그를 그대로** 붙여주세요(가장 빠릅니다).
- 결과가 이상하면 “입력 텍스트 / 설정값 / 출력” 3가지를 같이 주세요.

## Windows 한글 출력이 깨지면

PowerShell에서 아래 중 하나를 먼저 실행해보세요:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
```
