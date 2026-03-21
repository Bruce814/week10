"""
Week17: RLHF — 基于奖励模型对语言模型进行强化学习 (PPO), 纯 PyTorch 从零实现.

运行: python tests/week17/3_rlhf_llm_ppo.py
依赖: pip install torch matplotlib

==========================================================================
完整 RLHF 三阶段, 本文件一次性演示:

  阶段 1: Pre-train  — 在混合正负面语料上预训练一个 Tiny GPT
  阶段 2: Reward Model — 定义奖励模型 (正面文本 -> 高分, 负面 -> 低分)
  阶段 3: PPO         — 用强化学习微调 GPT, 使其倾向生成正面文本

三个核心角色:
  +-----------------+     +-----------------+     +-----------------+
  |  Policy Model   |     | Reference Model |     |  Reward Model   |
  | pi_theta (train)|     | pi_ref (frozen) |     |  R(text) score  |
  +--------+--------+     +--------+--------+     +--------+--------+
           |                       |                       |
           | generate response     | KL divergence         | score response
           v                       v                       v
      +----------------------------------------------------------+
      | PPO Loss = -min(ratio*A, clip(ratio, 1+/-eps)*A)         |
      | ratio = pi_theta / pi_old                                |
      | advantage A = returns (reward - KL penalty)              |
      +----------------------------------------------------------+

为什么需要 Reference Model + KL 惩罚?
  只最大化 reward 会导致模型 "reward hacking"
  (例如不断重复正面词, 输出无意义文本).
  KL(pi_theta || pi_ref) 约束模型不要偏离原始分布太远.

本示例效果:
  预训练后: 模型生成正面和负面文本各占约 50%
  RLHF 后:  模型明显倾向生成正面文本, 平均奖励显著提升
==========================================================================
"""

import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 1. Vocabulary & Tokenizer
# ====================================================================

_SPECIAL = ["<pad>", "<bos>", "<eos>"]
_WORDS = [
    # function words
    "the", "a", "is", "was", "i", "my", "this", "it",
    "very", "really", "quite", "not", "and", "but",
    "have", "would", "think", "say", "feel",
    "of", "for", "in", "here", "about", "to",
    # topics
    "movie", "film", "food", "restaurant", "book", "hotel",
    "product", "service", "place", "experience", "quality",
    "staff", "story", "room", "view", "atmosphere",
    # positive
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "beautiful", "perfect", "brilliant", "outstanding", "lovely",
    "superb", "delightful", "pleasant", "impressive", "incredible",
    # negative
    "bad", "terrible", "awful", "horrible", "boring", "disappointing",
    "ugly", "poor", "dull", "mediocre", "unpleasant", "rude",
    "frustrating", "annoying", "overpriced", "cold",
    # verbs / misc
    "love", "hate", "enjoy", "like", "recommend", "avoid",
]


class Tokenizer:
    def __init__(self):
        tokens = _SPECIAL + sorted(set(_WORDS))
        self.w2i = {w: i for i, w in enumerate(tokens)}
        self.i2w = {i: w for w, i in self.w2i.items()}
        self.vocab_size = len(tokens)
        self.pad = self.w2i["<pad>"]
        self.bos = self.w2i["<bos>"]
        self.eos = self.w2i["<eos>"]

    def encode(self, text: str, add_bos=True, add_eos=True) -> list[int]:
        ids = [self.w2i[w] for w in text.lower().split() if w in self.w2i]
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids

    def decode(self, ids, skip_special=True) -> str:
        words = []
        for idx in ids:
            i = idx.item() if hasattr(idx, "item") else idx
            w = self.i2w.get(i, "?")
            if skip_special and w in _SPECIAL:
                continue
            words.append(w)
        return " ".join(words)


# ====================================================================
# 2. Tiny GPT Model (2-layer Transformer Decoder)
# ====================================================================

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, n_layers=2, max_len=20):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.blocks(h, mask=causal_mask)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=8, temperature=0.8, eos_id=2):
        ids = prompt_ids.clone()
        for _ in range(max_new):
            if ids.shape[1] >= self.max_len:
                break
            logits = self(ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == eos_id:
                break
        return ids


# ====================================================================
# 3. Pre-training Data
# ====================================================================

_POS_ADJ = [
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "beautiful", "perfect", "brilliant", "outstanding", "lovely",
    "superb", "delightful", "pleasant", "impressive", "incredible",
]
_NEG_ADJ = [
    "bad", "terrible", "awful", "horrible", "boring", "disappointing",
    "ugly", "poor", "dull", "mediocre", "unpleasant", "rude",
]
_TOPICS = ["movie", "food", "service", "book", "hotel", "product", "place", "experience"]

_TEMPLATES = [
    lambda t, a: f"the {t} was {a}",
    lambda t, a: f"the {t} was very {a}",
    lambda t, a: f"i think the {t} is {a}",
    lambda t, a: f"my {t} was really {a}",
    lambda t, a: f"this {t} is {a}",
    lambda t, a: f"the {t} here is quite {a}",
]


def make_corpus(n=600):
    corpus = []
    for _ in range(n):
        tmpl = random.choice(_TEMPLATES)
        topic = random.choice(_TOPICS)
        adj = random.choice(_POS_ADJ if random.random() < 0.5 else _NEG_ADJ)
        corpus.append(tmpl(topic, adj))
    return corpus


# ====================================================================
# 4. Pre-training (Causal LM)
# ====================================================================

def pretrain(model, tok, corpus, epochs=50, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    encoded = [tok.encode(s) for s in corpus]
    max_len = min(max(len(s) for s in encoded), model.max_len)
    for i in range(len(encoded)):
        encoded[i] = encoded[i][:max_len]
        encoded[i] += [tok.pad] * (max_len - len(encoded[i]))
    data = torch.tensor(encoded)

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(data))
        total_loss, n = 0.0, 0
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i + batch_size]]
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tok.pad,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if ep % 10 == 0:
            print(f"    Epoch {ep:>3d}/{epochs} | Loss: {total_loss / n:.4f}")


# ====================================================================
# 5. Reward Model
# ====================================================================

class RewardModel:
    """
    R(text): positive words -> +2, negative words -> -2.
    Simulates a reward model trained on human preferences.
    """
    POS = set(_POS_ADJ)
    NEG = set(_NEG_ADJ)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        scores = []
        for text in texts:
            words = text.lower().split()
            if not words:
                scores.append(0.0)
                continue
            # Unique positive/negative words (no benefit from repetition)
            unique = set(words)
            p = sum(1.0 for w in unique if w in self.POS)
            n = sum(1.0 for w in unique if w in self.NEG)
            scores.append((p - n) * 2.0)
        return torch.tensor(scores, dtype=torch.float32)


# ====================================================================
# 6. PPO Trainer for RLHF (core algorithm)
# ====================================================================

class RLHFTrainer:
    """
    Proximal Policy Optimization for language model alignment.

    Key idea: maximize reward while staying close to the reference model.
      Loss = -min(ratio * advantage, clip(ratio) * advantage)
      Total reward = R(text) - beta * KL(policy || reference)
    """

    def __init__(self, policy, tok, reward_fn,
                 lr=3e-4, kl_coef=1.0, clip_eps=0.2, ppo_epochs=3):
        self.policy = policy
        self.ref = deepcopy(policy).eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        self.tok = tok
        self.reward_fn = reward_fn
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.reward_baseline = 0.0

    def _log_probs(self, model, prompt_ids, resp_ids):
        """Per-token log P(response_token | prompt + preceding response tokens)."""
        full = torch.cat([prompt_ids, resp_ids], dim=1)
        logits = model(full)
        plen = prompt_ids.shape[1]
        rlen = resp_ids.shape[1]
        # logits[t] predicts token[t+1], so response[0] <- logits[plen-1]
        resp_logits = logits[:, plen - 1: plen - 1 + rlen, :]
        lp = F.log_softmax(resp_logits, dim=-1)
        return lp.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)  # [1, rlen]

    def step(self, prompt: str):
        # ---- 1. Generate ----
        prompt_tokens = self.tok.encode(prompt, add_bos=True, add_eos=False)
        prompt_ids = torch.tensor([prompt_tokens])

        self.policy.eval()
        full = self.policy.generate(
            prompt_ids, max_new=8, temperature=0.9, eos_id=self.tok.eos
        )
        self.policy.train()

        resp_ids = full[:, len(prompt_tokens):]
        if resp_ids.shape[1] == 0:
            return None
        resp_text = self.tok.decode(resp_ids[0])

        # ---- 2. Reward ----
        reward = self.reward_fn([resp_text])[0]

        # ---- 3. Old log probs (frozen) ----
        with torch.no_grad():
            old_lp = self._log_probs(self.policy, prompt_ids, resp_ids)
            ref_lp = self._log_probs(self.ref, prompt_ids, resp_ids)

        # ---- 4. PPO update ----
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward.item()
        advantage = reward - self.reward_baseline

        total_loss = 0.0
        kl_val = 0.0
        for _ in range(self.ppo_epochs):
            new_lp = self._log_probs(self.policy, prompt_ids, resp_ids)

            # Per-token importance sampling ratio
            ratio = torch.exp(new_lp.squeeze(0) - old_lp.squeeze(0))

            # Clipped surrogate: advantage is scalar (same for all tokens)
            s1 = ratio * advantage
            s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(s1, s2).mean()

            # KL penalty (always >= 0): penalizes any deviation from reference
            # Using squared log-ratio as a stable, non-negative KL proxy
            kl_penalty = 0.5 * ((new_lp - ref_lp) ** 2).mean()

            loss = policy_loss + self.kl_coef * kl_penalty

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            total_loss += loss.item()
            kl_val = kl_penalty.item()

        return {
            "loss": total_loss / self.ppo_epochs,
            "reward": reward.item(),
            "kl": kl_val,
            "text": resp_text,
        }


# ====================================================================
# 7. Evaluation
# ====================================================================

_EVAL_PROMPTS = [
    "the movie was", "the food was", "the service was",
    "my experience was", "this place is", "the hotel was",
    "the product is", "i think the book is",
]


def evaluate(model, tok, reward_fn, n=3):
    model.eval()
    all_r = []
    for prompt in _EVAL_PROMPTS:
        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        pid = torch.tensor([toks])
        for _ in range(n):
            full = model.generate(pid, max_new=8, temperature=0.7, eos_id=tok.eos)
            resp = tok.decode(full[0][len(toks):])
            r = reward_fn([resp])[0].item()
            all_r.append(r)
            print(f"    {prompt} ... {resp}  (reward={r:.1f})")
    avg = sum(all_r) / len(all_r)
    print(f"\n    Average Reward: {avg:.2f}")
    return avg


# ====================================================================
# 8. Plotting
# ====================================================================

def plot(reward_hist, window=20):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [skip plot] pip install matplotlib")
        return

    sm = []
    for i in range(len(reward_hist)):
        s = max(0, i - window + 1)
        sm.append(sum(reward_hist[s:i + 1]) / (i - s + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(reward_hist, alpha=0.3, color="steelblue", label="Per-step Reward")
    plt.plot(sm, color="darkorange", linewidth=2, label=f"Moving Avg ({window})")
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Neutral (0)")
    plt.xlabel("PPO Step")
    plt.ylabel("Reward")
    plt.title("RLHF Training - Reward over PPO Steps")
    plt.legend()
    plt.tight_layout()
    path = "tests/week17/rlhf_training_curve.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Training curve saved to {path}")


# ====================================================================
# 9. Main
# ====================================================================

def main():
    tok = Tokenizer()
    reward_fn = RewardModel()
    print(f"  Vocab size: {tok.vocab_size}")

    # -- Stage 1: Pre-train --
    corpus = make_corpus(n=800)
    print(f"  Corpus: {len(corpus)} sentences (50% positive, 50% negative)")

    model = TinyGPT(tok.vocab_size, d_model=64, nhead=4, n_layers=2, max_len=20)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: TinyGPT ({n_params:,} parameters)\n")

    print("  [Stage 1/2] Pre-training (Causal LM)...")
    pretrain(model, tok, corpus, epochs=50, lr=1e-3)

    print("\n  === Before RLHF ===")
    print("  " + "-" * 55)
    avg_before = evaluate(model, tok, reward_fn, n=2)

    # -- Stage 2: RLHF with PPO --
    trainer = RLHFTrainer(
        model, tok, reward_fn,
        lr=3e-4, kl_coef=0.6, clip_eps=0.2, ppo_epochs=3,
    )

    num_steps = 800
    prompts = [
        "the movie was", "the food was", "the service was",
        "my experience was", "this place is", "the hotel was",
        "i think the product is", "the book was",
    ]

    print(f"\n  [Stage 2/2] RLHF - PPO ({num_steps} steps)...\n")

    reward_hist = []
    for step in range(1, num_steps + 1):
        stats = trainer.step(random.choice(prompts))
        if stats is None:
            continue
        reward_hist.append(stats["reward"])

        if step % 100 == 0:
            rec = reward_hist[-100:]
            print(
                f"  Step {step:>3d}/{num_steps} | "
                f"Loss: {stats['loss']:.4f} | "
                f"Reward: {stats['reward']:>5.1f} | "
                f"Avg(100): {sum(rec) / len(rec):>5.2f} | "
                f"KL: {stats['kl']:.3f} | "
                f"Sample: {stats['text']!r}"
            )

    print("\n  === After RLHF ===")
    print("  " + "-" * 55)
    avg_after = evaluate(model, tok, reward_fn, n=2)

    # Summary
    delta = avg_after - avg_before
    print("\n  " + "=" * 55)
    print(f"  Average Reward  BEFORE RLHF:  {avg_before:>6.2f}")
    print(f"  Average Reward  AFTER  RLHF:  {avg_after:>6.2f}")
    print(f"  Improvement:                  {delta:>+6.2f}")
    print("  " + "=" * 55)

    if delta > 0:
        print("\n  [OK] RLHF shifted the model toward positive text generation!")
    else:
        print("\n  [INFO] Try more steps or tune hyperparameters.")

    plot(reward_hist)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 62)
    print("  Week17: RLHF - Reward Model + PPO for Language Model")
    print("=" * 62)
    print("""
  Full pipeline (pure PyTorch, no downloads):

    1. Pre-train a Tiny GPT on mixed-sentiment text
    2. Define a Reward Model (positive text -> high score)
    3. PPO fine-tune: maximize reward with KL constraint
    4. Compare before / after RLHF

  Key components:
    Policy Model  pi_theta  = trainable GPT
    Reference     pi_ref    = frozen copy (KL anchor)
    Reward Model  R(text)   = sentiment scorer
    """)

    main()

    print("\n" + "=" * 62)
    print("  Done!")
    print("=" * 62)
    print("""
  --- Key Takeaways ---
  * RLHF optimizes LLMs toward human preferences via reward model
  * PPO + KL penalty prevents "reward hacking" (degenerate outputs)
  * This demo: reward = positive words, but real RLHF uses human-
    preference-trained reward models (e.g. Bradley-Terry model)

  --- Production Tools ---
  * trl (HuggingFace):  pip install trl  — PPO/DPO/GRPO trainers
  * DPO (Direct Preference Optimization): no reward model needed
  * GRPO (DeepSeek R1): group relative policy optimization
    """)
