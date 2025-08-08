#!/usr/bin/env python3
import os
import time
import json
import torch
from pathlib import Path
from quantum_gpt_implementation import (
    QuantumGPT, ClassicalGPT, get_batch,
    n_embd, n_head, n_layer, block_size, batch_size, learning_rate
)

DATA_PATH = Path("data/input.txt")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = MODELS_DIR / "results.md"

DEVICE = torch.device("cpu")
MAX_ITERS = 200
EVAL_INTERVAL = 50
EVAL_ITERS = 20
GEN_TOKENS = 120


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/input.txt not found. Download tiny Shakespeare before running.")
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars), encode, decode


def train_model(model, train_data):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.time()

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(EVAL_ITERS):
                    X, Y = get_batch(train_data, batch_size, block_size)
                    X, Y = X.to(DEVICE), Y.to(DEVICE)
                    _, loss = model(X, Y)
                    losses.append(loss.item())
                avg_loss = float(torch.tensor(losses).mean().item())
            model.train()
        X, Y = get_batch(train_data, batch_size, block_size)
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    total_time = time.time() - start
    return avg_loss, total_time


def greedy_generate(model, decode, prompt=""):
    device = next(model.parameters()).device
    if prompt:
        raise NotImplementedError("Deterministic prompt encoding not supported in this script.")
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(GEN_TOKENS):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return decode(idx[0].tolist())


def main():
    torch.manual_seed(0)
    train_data, val_data, vocab_size, encode, decode = load_dataset()

    meta = {
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(DEVICE),
        "max_iters": MAX_ITERS,
        "eval_iters": EVAL_ITERS,
    }

    # Classical
    classical = ClassicalGPT(vocab_size)
    classical_loss, classical_time = train_model(classical, train_data)
    classical_path = MODELS_DIR / "classical_gpt.pth"
    torch.save({
        "model_state_dict": classical.state_dict(),
        "vocab_size": vocab_size,
        "model_type": "classical",
        "meta": meta,
    }, classical_path)
    classical_out = greedy_generate(classical, decode)

    # Quantum
    quantum = QuantumGPT(vocab_size)
    quantum_loss, quantum_time = train_model(quantum, train_data)
    quantum_path = MODELS_DIR / "quantum_gpt.pth"
    torch.save({
        "model_state_dict": quantum.state_dict(),
        "vocab_size": vocab_size,
        "model_type": "quantum",
        "meta": meta,
    }, quantum_path)
    quantum_out = greedy_generate(quantum, decode)

    # Write results markdown
    lines = []
    lines.append("# Training Results\n")
    lines.append("\n## Configuration\n")
    lines.append(f"- n_embd: {n_embd}")
    lines.append(f"- n_head: {n_head}")
    lines.append(f"- n_layer: {n_layer}")
    lines.append(f"- block_size: {block_size}")
    lines.append(f"- batch_size: {batch_size}")
    lines.append(f"- learning_rate: {learning_rate}")
    lines.append(f"- device: {DEVICE}")
    lines.append(f"- max_iters: {MAX_ITERS}")
    lines.append(f"- eval_iters: {EVAL_ITERS}\n")

    lines.append("\n## Results Summary\n")
    lines.append(f"- Classical loss: {classical_loss:.4f} (time: {classical_time:.1f}s)")
    lines.append(f"- Quantum loss: {quantum_loss:.4f} (time: {quantum_time:.1f}s)\n")

    lines.append("\n## Sample Outputs\n")
    lines.append("### Classical GPT\n")
    lines.append("```\n" + classical_out + "\n```\n")
    lines.append("### Quantum GPT\n")
    lines.append("```\n" + quantum_out + "\n```\n")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved classical to {classical_path}")
    print(f"Saved quantum to {quantum_path}")
    print(f"Wrote report to {RESULTS_MD}")


if __name__ == "__main__":
    main() 