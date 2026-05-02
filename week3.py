"""Week 3 — 빅램(2-gram) 카운트 언어모델.

이 주에 배우는 것
- 토큰화(tokenization): 텍스트를 숫자 ID 의 시퀀스로 바꾸는 일.
- 가장 단순한 언어모델: "글자 a 다음에 글자 b 가 몇 번 나왔는지" 를 세서 확률표를 만든다.
- 그 확률표로 새 텍스트를 한 글자씩 샘플링.

왜 이게 필요한가
- 신경망 없이도 "언어모델" 을 만들 수 있다는 걸 직접 본다.
- 다음 주에 똑같은 일을 신경망으로 다시 하면, 신경망이 뭘 자동화해주는지 명확해진다.

실행:  python week3.py
"""

import numpy as np


# ============================================================
# 0. 학습 데이터 — 짧은 한국어 인사 모음
# ============================================================
# 진짜 LLM 은 인터넷 전체를 읽지만, 우리는 손으로 셀 수 있는 작은 코퍼스를 쓴다.
# 같은 표현이 반복되어야 "패턴" 이 통계로 잡힌다.

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
)


# ============================================================
# 1. 토큰화 — 글자 단위 (character-level)
# ============================================================
# 가장 단순한 방법: 등장한 글자 하나하나에 0, 1, 2, ... 번호를 매긴다.
# 진짜 LLM 은 BPE(서브워드) 를 쓰지만 (week10 에서 살짝 본다), 원리는 같다.
#
# 자료구조:
#   id_to_char : list[str]   ← index 가 ID, 값이 글자
#   char_to_id : dict[str, int]   ← Java 의 HashMap<String, Integer>
#
# 정렬한 set 을 쓰면 같은 텍스트면 항상 같은 ID 가 나온다 (재현성).

class CharTokenizer:
    def __init__(self, text):
        # 등장한 글자만 모아서 정렬. set() 은 Java 의 HashSet.
        unique_chars = sorted(set(text))
        self.id_to_char = unique_chars

        # dict comprehension 안 쓰고 명시적 for.
        self.char_to_id = {}
        for i in range(len(unique_chars)):
            self.char_to_id[unique_chars[i]] = i

    def vocab_size(self):
        return len(self.id_to_char)

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            chars.append(self.id_to_char[token_id])
        return "".join(chars)


tokenizer = CharTokenizer(TRAINING_TEXT)
vocab_size = tokenizer.vocab_size()

print("=" * 60)
print("[1] 토큰화")
print("=" * 60)
print("vocab size =", vocab_size)
print("처음 10 글자 ID:", tokenizer.id_to_char[:10])
print("'안녕' encode :", tokenizer.encode("안녕"))
print("decode 다시  :", tokenizer.decode(tokenizer.encode("안녕")))


# ============================================================
# 2. 빅램 카운트 행렬 만들기
# ============================================================
# bigram(2-gram) = 연속된 두 글자.
# counts[a][b] = 글자 a 다음에 글자 b 가 나온 횟수.
#
# Java 비유: int[V][V] 2차원 배열에 ++ 한다고 보면 됨.
# (V = vocab_size)

print()
print("=" * 60)
print("[2] 빅램 카운트")
print("=" * 60)

# numpy float 행렬. 나중에 확률로 나눌 때 편하다.
counts = np.zeros((vocab_size, vocab_size))

# 텍스트 전체를 ID 시퀀스로 바꾸고
ids = tokenizer.encode(TRAINING_TEXT)

# 인접한 두 글자 쌍을 모두 돌면서 카운트 ++
for i in range(len(ids) - 1):
    prev_id = ids[i]
    next_id = ids[i + 1]
    counts[prev_id][next_id] += 1

print("counts shape =", counts.shape)
print("총 bigram 개수 =", int(counts.sum()))

# '안' 다음에 가장 많이 나온 글자 top 3 보기
target_char = "안"
target_id = tokenizer.char_to_id[target_char]
row = counts[target_id]
# row 에서 가장 큰 인덱스 3개를 직접 찾는다 (argsort 한 번만 사용).
top3_indices = np.argsort(row)[::-1][:3]
print("'%s' 다음 top3 :" % target_char)
for idx in top3_indices:
    if row[idx] > 0:
        print("  '%s' = %d번" % (tokenizer.id_to_char[idx], int(row[idx])))


# ============================================================
# 3. 카운트 → 확률표
# ============================================================
# 각 행을 그 행의 합으로 나누면 확률이 된다.
#   probs[a][b] = P(b | a) = counts[a][b] / sum_k counts[a][k]
#
# 한 번도 등장 안 한 행이 있으면 0/0 이 되니까 작은 수(smoothing)를 더한다.

# 라플라스 스무딩: 모든 셀에 +1 (한 번도 안 본 조합도 아주 작은 확률을 갖게)
counts_smoothed = counts + 1.0

# 행 합 (broadcasting 을 쓰기 위해 keepdims=True)
row_sums = counts_smoothed.sum(axis=1, keepdims=True)   # shape (V, 1)
probs = counts_smoothed / row_sums                       # shape (V, V)

# 각 행의 합이 1 인지 확인
print()
print("[3] 확률표 만들기")
print("각 행 합이 1 인지 확인 (처음 5행):", probs.sum(axis=1)[:5])


# ============================================================
# 4. 샘플링 — 확률표에서 다음 글자 뽑기
# ============================================================
# 현재 글자 c 가 주어지면 probs[c] 가 다음 글자의 확률 분포다.
# numpy 의 random.choice 가 분포대로 뽑아준다.
#
# Java 라면: 0~1 난수 뽑고, 누적합 따라가서 어느 구간에 떨어지는지 찾기.

def sample_next_id(current_id, probs, rng):
    distribution = probs[current_id]
    # rng.choice(N, p=...) : 0..N-1 중 p 분포 따라 하나 뽑기
    return int(rng.choice(len(distribution), p=distribution))


def generate(seed_text, num_chars, probs, tokenizer, rng):
    # 시드(seed) 텍스트로 시작해서 한 글자씩 이어 붙인다.
    output = list(seed_text)

    # 마지막 글자가 다음 예측의 입력
    current_id = tokenizer.char_to_id[seed_text[-1]]

    for _ in range(num_chars):
        next_id = sample_next_id(current_id, probs, rng)
        next_char = tokenizer.id_to_char[next_id]
        output.append(next_char)
        current_id = next_id

    return "".join(output)


# ============================================================
# 5. 실제 생성
# ============================================================
print()
print("=" * 60)
print("[5] 텍스트 생성")
print("=" * 60)

# 같은 결과 재현용 난수 생성기
rng = np.random.default_rng(42)

for trial in range(3):
    text = generate(seed_text="안녕", num_chars=40, probs=probs,
                    tokenizer=tokenizer, rng=rng)
    print("샘플 %d: %s" % (trial + 1, text))


# ============================================================
# 6. greedy vs sampling
# ============================================================
# greedy = 항상 확률이 가장 높은 글자만 고르기. 같은 시드면 항상 같은 결과.
# 보통 짧게 가다가 같은 패턴에 갇힌다.

def generate_greedy(seed_text, num_chars, probs, tokenizer):
    output = list(seed_text)
    current_id = tokenizer.char_to_id[seed_text[-1]]
    for _ in range(num_chars):
        # argmax = 가장 큰 값의 인덱스
        next_id = int(np.argmax(probs[current_id]))
        output.append(tokenizer.id_to_char[next_id])
        current_id = next_id
    return "".join(output)

print()
print("[6] greedy(항상 최고확률) 결과:")
print(generate_greedy("안녕", 40, probs, tokenizer))
print()
print("→ greedy 는 보통 같은 글자가 반복된다. 그래서 다양성을 위해 샘플링을 쓴다.")


# ============================================================
# 7. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- 언어모델 = '다음 토큰 확률 분포' 를 출력하는 함수.")
print("- 빅램 카운트 모델은 학습이 곧 카운팅이다 (경사하강법 없음).")
print("- 한계: 직전 1글자만 보니 '안녕하세' 같은 4글자 문맥은 못 본다.")
print("- 다음 주: 같은 일을 신경망(임베딩 + softmax) 으로 다시.")
