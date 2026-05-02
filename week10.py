"""Week 10 — 다음 단계: PyTorch 입문 + BPE / SFT 미리보기.

이 주에 배우는 것
- 같은 MiniGPT(week9) 를 PyTorch 로 다시 쓰면서 "프레임워크가 뭘 자동화해주는지" 체감.
  → autograd: 미분을 자동으로. 우리는 forward 만 쓴다.
  → optimizer: SGD/Adam 같은 갱신 규칙도 한 줄.
  → nn.Module: 가중치 묶음을 클래스로 깔끔하게.
- BPE(Byte Pair Encoding) 토크나이저가 무엇이고 왜 필요한지.
- SFT(Supervised Fine-Tuning) 데이터가 어떻게 생겼는지.

PyTorch 가 없으면?
- 자동으로 numpy 부분만 실행되고 PyTorch 부분은 건너뛴다.

실행:  python week10.py
"""

import json


# ============================================================
# 0. PyTorch 설치 확인
# ============================================================
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# 1. PyTorch — autograd 5분 입문
# ============================================================
# numpy 와의 가장 큰 차이 한 줄:
#   "tensor 에 requires_grad=True 를 주면, 모든 연산이 그래프에 기록된다."
#   "loss.backward() 한 번 부르면 모든 텐서의 .grad 가 자동으로 채워진다."
#
# week5 에서 손으로 유도했던 d(loss)/d(z) 같은 미분식을 직접 안 써도 된다.

def demo_autograd():
    print("=" * 60)
    print("[1] autograd: 손미분 vs torch 자동미분")
    print("=" * 60)

    # 같은 함수 f(x) = x^2 + 3x + 5 를 두 가지로 미분
    # 손 미분:  f'(x) = 2x + 3
    # x = 4 에서  f'(4) = 11

    x = torch.tensor(4.0, requires_grad=True)
    y = x * x + 3 * x + 5
    y.backward()           # 자동으로 미분
    print("torch 자동: dy/dx at x=4 =", x.grad.item())
    print("손 미분    : 2*4 + 3 =", 2 * 4 + 3)


# ============================================================
# 2. nn.Module — 가중치 묶음을 클래스로
# ============================================================
# week9 에서 우리는 dict 로 params 를 들고 다녔다.
# PyTorch 에서는 nn.Module 을 상속받아 __init__ 에서 가중치를 선언한다.
# Java 비유: 가중치는 클래스 필드(field), forward 는 메서드.
#
# 더 좋은 점: 모든 가중치 갱신을 optimizer 한 줄로 처리.

class MiniGPT(nn.Module if HAS_TORCH else object):
    """week9 와 동일한 구조: token+pos emb, attention(1 head), 잔차, FFN, 잔차, 출력."""

    def __init__(self, vocab_size, seq_len, d_model, ffn_hidden):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # nn.Embedding = E[id] 형태의 학습 가능한 lookup 테이블
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # nn.Linear = (W, b) 묶음. y = x @ W.T + b 형태로 동작 (PyTorch 관례).
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # 이번에는 LayerNorm 도 추가 (week9 에서 단순화 위해 뺐었던 것).
        # 진짜 GPT 가 쓰는 형태에 한 발 더 가까워진다.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, ffn_hidden)
        self.ffn2 = nn.Linear(ffn_hidden, d_model)

        self.out_head = nn.Linear(d_model, vocab_size)

        # causal mask 를 buffer 로 (학습 안 되는 텐서, 모델과 함께 이동)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, ids):
        # ids shape (B, T)
        B = ids.shape[0]
        T = ids.shape[1]

        positions = torch.arange(T, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(positions)            # (B, T, D)

        # ---- Attention sub-layer ----
        x_norm = self.ln1(x)
        Q = self.W_Q(x_norm)                                         # (B, T, D)
        K = self.W_K(x_norm)
        V = self.W_V(x_norm)

        # scores: (B, T, T). 마지막 두 축에서 행렬곱.
        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn_raw = weights @ V                                       # (B, T, D)
        attn_out = self.W_O(attn_raw)
        x = x + attn_out                                             # 잔차 1

        # ---- FFN sub-layer ----
        x_norm = self.ln2(x)
        ffn_out = self.ffn2(torch.relu(self.ffn1(x_norm)))
        x = x + ffn_out                                              # 잔차 2

        logits = self.out_head(x)                                    # (B, T, V)
        return logits


# ============================================================
# 3. PyTorch 로 학습 루프 (numpy 버전과 비교용)
# ============================================================
def train_minigpt_torch():
    print()
    print("=" * 60)
    print("[3] PyTorch 로 MiniGPT 학습")
    print("=" * 60)

    TRAINING_TEXT = (
        "안녕하세요. 오늘 날씨가 참 좋네요.\n"
        "안녕하세요. 식사는 하셨나요?\n"
        "안녕하세요. 만나서 반갑습니다.\n"
        "오늘 날씨가 정말 좋습니다.\n"
        "오늘 식사는 무엇을 드셨나요?\n"
        "내일은 비가 올 것 같습니다.\n"
        "내일 만나서 식사 같이 해요.\n"
        "감사합니다. 좋은 하루 되세요.\n"
        "감사합니다. 또 만나요.\n"
        "반갑습니다. 좋은 시간 보내세요.\n"
        "오늘은 좋은 하루 보내세요.\n"
        "내일도 좋은 시간 되세요.\n"
    )

    # ---- 토크나이저 (week3-9 와 동일) ----
    unique_chars = sorted(set(TRAINING_TEXT))
    char_to_id = {}
    id_to_char = unique_chars
    for i in range(len(unique_chars)):
        char_to_id[unique_chars[i]] = i
    V = len(unique_chars)
    print("vocab size =", V)

    text_ids = []
    for ch in TRAINING_TEXT:
        text_ids.append(char_to_id[ch])

    # ---- 데이터셋 ----
    SEQ_LEN = 12
    inputs = []
    targets = []
    i = 0
    while i + SEQ_LEN < len(text_ids):
        window_in = []
        window_out = []
        for k in range(SEQ_LEN):
            window_in.append(text_ids[i + k])
            window_out.append(text_ids[i + k + 1])
        inputs.append(window_in)
        targets.append(window_out)
        i = i + 1
    inputs_t = torch.tensor(inputs, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)
    print("sequences =", len(inputs))

    # ---- 모델 ----
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=V, seq_len=SEQ_LEN, d_model=32, ffn_hidden=64)

    # PyTorch 의 핵심 한 줄: optimizer
    # optimizer 가 알아서 "param ← param - lr * grad" 를 해준다.
    # Adam 은 SGD 보다 조금 더 똑똑한 갱신 규칙 (학습률을 파라미터별로 조절).
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 손실 함수: cross-entropy. PyTorch 에 내장.
    loss_fn = nn.CrossEntropyLoss()

    # ---- 학습 ----
    NUM_EPOCHS = 200
    BATCH_SIZE = 16

    for epoch in range(NUM_EPOCHS):
        # 데이터 셔플
        perm = torch.randperm(len(inputs_t))
        x_shuf = inputs_t[perm]
        y_shuf = targets_t[perm]

        epoch_loss = 0.0
        num_batches = 0
        i = 0
        while i < len(x_shuf):
            X_batch = x_shuf[i : i + BATCH_SIZE]
            Y_batch = y_shuf[i : i + BATCH_SIZE]

            # ----- 핵심 4 줄 -----
            logits = model(X_batch)                                  # forward
            # CrossEntropyLoss 는 (N, V) 와 (N,) 모양을 기대한다.
            loss = loss_fn(
                logits.reshape(-1, V),
                Y_batch.reshape(-1),
            )
            optimizer.zero_grad()                                    # 누적 grad 비우기
            loss.backward()                                          # 자동 미분
            optimizer.step()                                         # 가중치 갱신
            # ---------------------

            epoch_loss += loss.item()
            num_batches += 1
            i += BATCH_SIZE

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / num_batches
            print("epoch %3d  avg loss = %.4f  ppl = %.2f"
                  % (epoch + 1, avg, float(torch.exp(torch.tensor(avg)).item())))

    # ---- 샘플링 ----
    print()
    print("[샘플링]")
    model.eval()
    with torch.no_grad():
        seed = "안녕하세요"
        generated = []
        for ch in seed:
            generated.append(char_to_id[ch])

        for _ in range(40):
            context = generated[-SEQ_LEN:]
            while len(context) < SEQ_LEN:
                context = [0] + context
            X_input = torch.tensor([context], dtype=torch.long)
            logits = model(X_input)                                  # (1, T, V)
            last_logits = logits[0, -1] / 1.0                        # temperature=1.0
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)

        out_text = []
        for token_id in generated:
            out_text.append(id_to_char[token_id])
        print("".join(out_text))


# ============================================================
# 4. BPE 미리보기 — 왜 글자 단위는 한계가 있는가
# ============================================================
# 우리는 지금까지 character-level (글자 단위) 토크나이저를 썼다.
# 장점: 단순, vocab 작음, 모르는 단어 없음.
# 단점:
#   - 시퀀스가 너무 길어진다. "안녕하세요" = 5 토큰. 같은 의미를 wordpiece 면 1~2개.
#   - 모델이 "글자 → 단어 의미" 까지 다시 학습해야 한다.
#
# 실제 LLM 은 BPE(Byte Pair Encoding) 를 쓴다.
#   1) 글자 단위로 시작.
#   2) 가장 자주 함께 등장하는 두 토큰을 하나로 합쳐서 새 토큰으로 만든다.
#   3) 정해진 vocab 크기에 도달할 때까지 반복.
#
# 결과: 자주 쓰는 단어/접미사는 한 토큰, 희귀 단어는 글자 조각으로.
#
# 진짜 학습은 안 하고 한 번의 merge 만 손으로 보여준다.

def bpe_one_step_demo():
    print()
    print("=" * 60)
    print("[4] BPE 한 스텝 시연")
    print("=" * 60)

    # 단어 단위로 빈도 (실제론 텍스트 전체에서)
    words = ["안녕하세요", "안녕하세요", "안녕히가세요", "감사합니다", "안녕"]

    # 일단 글자 토큰 시퀀스로 표현
    sequences = []
    for word in words:
        seq = []
        for ch in word:
            seq.append(ch)
        sequences.append(seq)

    print("초기 토큰화:")
    for s in sequences:
        print("  ", s)

    # 인접한 두 토큰 쌍의 등장 횟수 세기
    pair_counts = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1

    # 가장 많이 등장한 쌍 찾기
    best_pair = None
    best_count = 0
    for pair in pair_counts:
        if pair_counts[pair] > best_count:
            best_count = pair_counts[pair]
            best_pair = pair

    print()
    print("가장 자주 함께 등장한 쌍:", best_pair, "= %d번" % best_count)
    print("→ 이 쌍을 새 토큰 '%s%s' 로 합친다." % (best_pair[0], best_pair[1]))

    # 적용 (한 번)
    merged_token = best_pair[0] + best_pair[1]
    new_sequences = []
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                new_seq.append(merged_token)
                i = i + 2
            else:
                new_seq.append(seq[i])
                i = i + 1
        new_sequences.append(new_seq)

    print()
    print("merge 1번 후:")
    for s in new_sequences:
        print("  ", s)

    print()
    print("→ 이걸 N번 반복하면 BPE 학습 끝. 진짜 GPT 도 같은 알고리즘이다.")
    print("   (실전: tiktoken, sentencepiece 같은 라이브러리가 빠르게 처리)")


# ============================================================
# 5. SFT 데이터 미리보기
# ============================================================
# Pre-training (사전학습): 인터넷 크롤링한 raw 텍스트로 다음 토큰 예측만.
#   → "지식은 많지만 명령은 잘 따르지 않는" base model.
# Post-training (사후학습) 의 첫 단계 = SFT(Supervised Fine-Tuning):
#   → 사람이 만든 (질문, 좋은 답변) 쌍으로 추가 학습.
#   → "도움이 되는 답을 하는 챗봇" 으로 변신.
#
# 데이터 포맷은 보통 JSONL (한 줄에 JSON 1개).
# 가장 단순한 형태:
#   {"prompt": "...", "completion": "..."}
# OpenAI / Anthropic 의 채팅 포맷:
#   {"messages": [{"role": "user", "content": "..."},
#                 {"role": "assistant", "content": "..."}]}

def sft_data_demo():
    print()
    print("=" * 60)
    print("[5] SFT 데이터 형식 미리보기")
    print("=" * 60)

    # 가짜 SFT 예시 3개
    examples = [
        {
            "messages": [
                {"role": "user", "content": "오늘 기분이 어때?"},
                {"role": "assistant", "content": "도와드릴 준비가 됐어요."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "한 줄 코드로 1부터 10까지 더해줘 (Python)."},
                {"role": "assistant", "content": "sum(range(1, 11))"},
            ]
        },
        {
            "prompt": "회의록 핵심을 3줄로 요약해줘.",
            "completion": "1) ... 2) ... 3) ...",
        },
    ]

    print("JSONL 한 줄 = 학습 예시 1 개:")
    for ex in examples:
        print("  ", json.dumps(ex, ensure_ascii=False))

    # 간단한 검증: messages 형식이면 user/assistant 둘 다 있어야 의미가 있다.
    print()
    print("형식 검증:")
    for idx in range(len(examples)):
        ex = examples[idx]
        if "messages" in ex:
            roles = []
            for m in ex["messages"]:
                roles.append(m["role"])
            ok = ("user" in roles) and ("assistant" in roles)
            print("  예시 %d: 'messages' 형식, roles=%s, %s"
                  % (idx + 1, roles, "OK" if ok else "BAD"))
        elif "prompt" in ex and "completion" in ex:
            ok = len(ex["prompt"]) > 0 and len(ex["completion"]) > 0
            print("  예시 %d: 'prompt/completion' 형식, %s"
                  % (idx + 1, "OK" if ok else "BAD"))
        else:
            print("  예시 %d: 알 수 없는 형식" % (idx + 1))

    print()
    print("실제 SFT 학습 흐름:")
    print("  1) base model 불러오기")
    print("  2) 위 같은 (질문, 답) 쌍 수만 ~ 수십만 개 모으기")
    print("  3) 답변 부분에서만 cross-entropy loss 계산해 학습")
    print("  4) 평가 → 배포")
    print()
    print("→ 우리가 week9 까지 만든 'next-token loss' 가 그대로 쓰인다.")
    print("   유일하게 추가되는 것은 '입력 부분은 loss 에서 제외' 하는 mask 정도.")


# ============================================================
# 6. main
# ============================================================
def main():
    if HAS_TORCH:
        demo_autograd()
        train_minigpt_torch()
    else:
        print("[!] torch 가 설치되어 있지 않아 PyTorch 데모는 건너뜁니다.")
        print("    설치:  pip install torch")

    # 토크나이저/SFT 데모는 numpy 도 필요 없음
    bpe_one_step_demo()
    sft_data_demo()


main()


# ============================================================
# 7. 정리 — 10주 회고
# ============================================================
print()
print("=" * 60)
print("[10주 회고]")
print("=" * 60)
print("- W1~W2: numpy 와 학습의 기본")
print("- W3~W6: 언어모델을 차근차근 (카운트 → 신경망 → 컨텍스트 → MLP)")
print("- W7~W9: Transformer 의 모든 부품 → 학습 → 샘플링까지")
print("- W10  : 같은 모델을 PyTorch 로 → 진짜 LLM 의 다음 단계 (BPE / SFT)")
print()
print("이제 이런 것들을 직접 실험할 준비가 됐다:")
print("  - HuggingFace transformers 로 더 큰 모델 다루기")
print("  - 자기 데이터로 SFT 또는 LoRA 파인튜닝")
print("  - 평가 지표 (perplexity, MMLU 등) 로 모델 품질 측정")
print("  - 추론 최적화 (quantization, KV cache, batching)")
print()
print("핵심 메시지:")
print("  LLM 은 마법이 아니다. 행렬곱 + softmax + cross-entropy 의 반복일 뿐이다.")
