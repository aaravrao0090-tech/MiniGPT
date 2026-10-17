# MiniGPT 🧠

A tiny GPT-style language model trained from scratch. It uses character-level tokenization, causal self-attention, transformer blocks, and next-character prediction. It is a real neural model, not a wrapper around an online AI API.

## Install

Python 3.10+ recommended. In a terminal in this repository:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install PyTorch:
```bash
pip install -r requirements.txt
```

## Train

Edit `input.txt` and add text you have permission to use. Then run:

```bash
python train.py
```

The trained weights are saved to `minigpt.pt`.

## Generate text

After training, run:

```bash
python train.py --generate "MiniGPT"
```

Train longer with:
```bash
python train.py --steps 2000
```

## What to expect

The included dataset is only a tiny demo, so the model will produce limited, often nonsensical text. Use a larger clean text file for better experiments. This is a learning project, not a replacement for ChatGPT. It runs on CPU and uses CUDA automatically when available. No API key is needed.

## Files

- `train.py`: tokenizer, transformer architecture, training loop, text generation.
- `input.txt`: starter training corpus.
- `requirements.txt`: dependency list.
