"""MiniGPT: a tiny character-level GPT trained from scratch."""
import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

def build_model(vocab_size, block_size, n_embd=96, n_head=4, n_layer=3, dropout=0.1):
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            B, T, C = x.shape
            k, q = self.key(x), self.query(x)
            weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            weights = self.dropout(F.softmax(weights, dim=-1))
            return self.dropout(weights @ self.value(x))

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            head_size = n_embd // n_head
            self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(),
                                     nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
        def forward(self, x): return self.net(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa = MultiHeadAttention()
            self.ffwd = FeedForward()
            self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            return x + self.ffwd(self.ln2(x))

    class GPTLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.apply(self._init_weights)
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
            logits = self.lm_head(self.ln_f(self.blocks(x)))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            return logits, loss
        @torch.no_grad()
        def generate(self, idx, max_new_tokens=300, temperature=0.8):
            self.eval()
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 0.05)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    return GPTLanguageModel()

def main():
    parser = argparse.ArgumentParser(description="Train and run a tiny GPT-style language model.")
    parser.add_argument("--steps", type=int, default=1000, help="training steps (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--generate", type=str, default=None, help="load saved model and continue this prompt")
    parser.add_argument("--new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--data", type=str, default="input.txt")
    parser.add_argument("--save", type=str, default="minigpt.pt")
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}. Create a text file first.")
    text = open(args.data, "r", encoding="utf-8").read()
    if len(text) < 100:
        raise SystemExit("Training text is very short. Add at least 100 characters to input.txt.")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[int(i)] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    split = max(1, int(0.9 * len(data)))
    train_data, val_data = data[:split], data[split:]
    if len(train_data) <= args.block_size + 1:
        args.block_size = max(8, min(args.block_size, len(train_data) // 2))
    model = build_model(len(chars), args.block_size).to(device)
    if args.generate:
        if not os.path.exists(args.save):
            raise SystemExit(f"No saved model at {args.save}. Train first with: python train.py")
        checkpoint = torch.load(args.save, map_location=device, weights_only=False)
        if checkpoint["chars"] != chars or checkpoint["block_size"] != args.block_size:
            raise SystemExit("Dataset vocabulary or block size changed since training. Retrain with python train.py.")
        model.load_state_dict(checkpoint["model"])
        prompt = args.generate
        encoded = encode(prompt)
        if not encoded: encoded = [0]
        result = model.generate(torch.tensor([encoded], device=device), args.new_tokens, args.temperature)
        print(decode(result[0].tolist()))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    def get_batch(source):
        if len(source) <= args.block_size + 1:
            source = train_data
        ix = torch.randint(len(source) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([source[i:i+args.block_size] for i in ix])
        y = torch.stack([source[i+1:i+args.block_size+1] for i in ix])
        return x.to(device), y.to(device)
    eval_interval = max(1, args.steps // 10)
    model.train()
    for step in range(args.steps):
        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_interval == 0 or step == args.steps - 1:
            print(f"step {step+1:>5}/{args.steps} | loss {loss.item():.3f} | device {device}")
    torch.save({"model": model.state_dict(), "chars": chars, "block_size": args.block_size}, args.save)
    print(f"\nSaved model to {args.save}")
    prompt = "MiniGPT"
    encoded = encode(prompt)
    result = model.generate(torch.tensor([encoded or [0]], device=device), 250, 0.8)
    print("\n--- Sample generation ---\n" + decode(result[0].tolist()))

if __name__ == "__main__":
    main()
