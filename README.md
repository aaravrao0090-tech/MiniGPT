# MiniGPT v2 🧠

A small GPT-style language model built with PyTorch. This version uses a **byte-level tokenizer**, causal self-attention, residual transformer blocks, validation loss reporting, gradient clipping, top-k sampling, and saved checkpoints. It is a neural language model trained from scratch, not an API wrapper.

## Quick start

Python 3.10+ recommended.

\`\`\`powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
\`\`\`

The default configuration has 4 transformer blocks, 4 attention heads, and 128-dimensional embeddings. It trains for 2,000 steps and saves \`minigpt.pt\`.

## Generate text

After training:

\`\`\`powershell
python train.py --generate "Once upon a time"
\`\`\`

Adjust length and randomness:

\`\`\`powershell
python train.py --generate "Hello" --new-tokens 500 --temperature 0.7
\`\`\`

## Train on more data

Replace \`input.txt\` with a larger UTF-8 text file that you have permission to use. The model learns only from the data you give it. More varied, high-quality text generally helps, but this small model and training setup have limited capacity.

\`\`\`powershell
python train.py --data input.txt --steps 10000 --batch-size 32
\`\`\`

You can change the model size, for example:

\`\`\`powershell
python train.py --steps 5000 --n-layer 6 --n-embd 192 --n-head 6 --block-size 128
\`\`\`

Larger settings require more memory and time. The model automatically uses CUDA if PyTorch detects a compatible GPU; otherwise it runs on CPU.

## Important limitations

- The included sample dataset is intentionally tiny, so initial output will be weak or nonsensical.
- This is a character-byte predictor, not an instruction-tuned assistant. It does not inherently know facts, follow chat roles, or reason like a large pretrained model.
- Each time you train, the model starts from scratch. Generation loads the saved checkpoint.
- Training data must be at least 64 bytes. The context size is automatically reduced for small datasets.

## Project files

- \`train.py\`: transformer, training loop, validation, checkpointing, and generation.
- \`input.txt\`: small starter corpus. Replace it with your own larger dataset.
- \`requirements.txt\`: PyTorch dependency.
