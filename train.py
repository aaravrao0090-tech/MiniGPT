"""MiniGPT v2: a byte-level GPT-style language model built with PyTorch.

Train:    python train.py --steps 3000
Generate: python train.py --generate "Once upon a time"
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

VOCAB_SIZE = 256  # UTF-8 bytes: prompts need no fitted vocabulary.


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, t, _ = x.shape
        weights = self.query(x) @ self.key(x).transpose(-2, -1) * (self.key.out_features ** -0.5)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return self.dropout(weights @ self.value(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // n_head, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(VOCAB_SIZE, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(
            torch.arange(t, device=idx.device)
        )
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 0.05)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


def main():
    parser = argparse.ArgumentParser(description="Train a small GPT-style language model.")
    parser.add_argument("--data", default="input.txt", help="UTF-8 text training file")
    parser.add_argument("--save", default="minigpt.pt", help="checkpoint output path")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--generate", default=None, help="generate from a prompt using saved weights")
    parser.add_argument("--new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be positive.")
    if args.n_embd % args.n_head:
        raise SystemExit("--n-embd must be divisible by --n-head.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    if not os.path.exists(args.data):
        raise SystemExit(f"Can't find {args.data}. Add a UTF-8 text file and retry.")
    raw = open(args.data, "rb").read()
    if len(raw) < 64:
        raise SystemExit("Training data is too short. Add more text to the data file.")
    data = torch.tensor(list(raw), dtype=torch.long)
    block_size = min(args.block_size, max(8, len(data) // 10))
    if args.generate is not None:
        if not os.path.exists(args.save):
            raise SystemExit("No checkpoint found. Train first with: python train.py")
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = MiniGPT(**cfg).to(device)
        model.load_state_dict(ckpt["model"])
        prompt = torch.tensor([list(args.generate.encode("utf-8")) or [32]], device=device)
        out = model.generate(prompt, args.new_tokens, args.temperature)
        print(out[0].tolist()[len(prompt[0]):] if False else bytes(out[0].tolist()).decode("utf-8", errors="replace"))
        return

    split = max(1, int(0.9 * len(data)))
    train_data = data[:split]
    val_data = data[split:] if len(data) - split > block_size + 1 else train_data
    cfg = dict(block_size=block_size, n_embd=args.n_embd, n_head=args.n_head,
               n_layer=args.n_layer, dropout=0.1)
    model = MiniGPT(**cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"MiniGPT v2 | {params:,} parameters | {device} | {len(data):,} training bytes")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def batch(source):
        ix = torch.randint(0, len(source) - block_size - 1, (args.batch_size,))
        x = torch.stack([source[i:i + block_size] for i in ix])
        y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate(source):
        model.eval()
        vals = []
        for _ in range(3):
            xb, yb = batch(source)
            _, loss = model(xb, yb)
            vals.append(loss.item())
        model.train()
        return sum(vals) / len(vals)

    interval = max(1, args.steps // 20)
    model.train()
    for step in range(args.steps):
        xb, yb = batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % interval == 0 or step + 1 == args.steps:
            print(f"step {step + 1:>5}/{args.steps} | train {loss.item():.3f} | val {estimate(val_data):.3f}")

    torch.save({"model": model.state_dict(), "config": cfg}, args.save)
    print(f"\nCheckpoint saved: {args.save}")
    prompt = torch.tensor([list(b"MiniGPT")], device=device)
    out = model.generate(prompt, 200, args.temperature)
    print("\n--- Sample generation ---\n" + bytes(out[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
