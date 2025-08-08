#!/usr/bin/env python3
import sys
import io
import time
import cProfile
import pstats
import resource
from pathlib import Path
import os
import torch
import tqdm

# Threading and CPU utilization controls (auto-utilize all cores; override with NUM_THREADS)
T = int(os.getenv("NUM_THREADS", os.cpu_count() or 8))
os.environ["OMP_NUM_THREADS"] = str(T)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(T)  # macOS Accelerate/vecLib
os.environ["OPENBLAS_NUM_THREADS"] = str(T)
os.environ["MKL_NUM_THREADS"] = str(T)
try:
    torch.set_num_threads(T)
    torch.set_num_interop_threads(max(1, T // 2))
except Exception:
    pass

# Ensure we can import from src/
sys.path.append(str(Path(__file__).parent / "src"))
from quantum_gpt_implementation import (
    QuantumGPT, ClassicalGPT, get_batch,
    n_embd, n_head, n_layer, block_size, batch_size, learning_rate
)

DATA_PATH = Path("data/input.txt")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = MODELS_DIR / "results.md"
PROFILE_CLASSICAL = MODELS_DIR / "profile_classical.txt"
PROFILE_QUANTUM = MODELS_DIR / "profile_quantum.txt"

DEVICE = torch.device("cpu")
MAX_ITERS = 10000
EVAL_INTERVAL = 50
EVAL_ITERS = 20
GEN_TOKENS = 120


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/input.txt not found. Run: make data")
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
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.time()
    for it in tqdm.tqdm(range(MAX_ITERS)):
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
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f"Iteration {it} loss: {avg_loss:.4f}")
    total_time = time.time() - start
    return avg_loss, total_time


def greedy_generate(model, decode, prompt=""):
    device = next(model.parameters()).device
    if prompt:
        raise NotImplementedError("Deterministic prompt encoding not supported.")
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(GEN_TOKENS):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return decode(idx[0].tolist())


def hamming_and_agreement(a_tokens, b_tokens):
    n = min(len(a_tokens), len(b_tokens))
    if n == 0:
        return 0, 0.0
    mismatches = sum(1 for i in range(n) if a_tokens[i] != b_tokens[i])
    agree = 1 - mismatches / n
    return mismatches, agree


def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not (A or B):
        return 1.0
    return len(A & B) / max(1, len(A | B))


def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev2 = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = prev2
    return dp[m]


def profile_block(callable_fn):
    pr = cProfile.Profile()
    pr.enable()
    t0_cpu = time.process_time()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    result = callable_fn()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    t1_cpu = time.process_time()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(25)
    prof_txt = s.getvalue()
    cpu_time = t1_cpu - t0_cpu
    max_rss_mb = r1.ru_maxrss / 1e6
    return result, prof_txt, cpu_time, max_rss_mb


def param_count(model):
    return sum(p.numel() for p in model.parameters())


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

    # Classical (profiled)
    classical = ClassicalGPT(vocab_size)
    (classical_loss, classical_time), prof_c, cpu_c, rss_c = profile_block(lambda: train_model(classical, train_data))
    classical_out = greedy_generate(classical, decode)
    classical_path = MODELS_DIR / "classical_gpt.pth"
    torch.save({
        "model_state_dict": classical.state_dict(),
        "vocab_size": vocab_size,
        "model_type": "classical",
        "meta": meta,
    }, classical_path)
    PROFILE_CLASSICAL.write_text(prof_c, encoding="utf-8")

    # Quantum (profiled)
    quantum = QuantumGPT(vocab_size)
    (quantum_loss, quantum_time), prof_q, cpu_q, rss_q = profile_block(lambda: train_model(quantum, train_data))
    quantum_out = greedy_generate(quantum, decode)
    quantum_path = MODELS_DIR / "quantum_gpt.pth"
    torch.save({
        "model_state_dict": quantum.state_dict(),
        "vocab_size": vocab_size,
        "model_type": "quantum",
        "meta": meta,
    }, quantum_path)
    PROFILE_QUANTUM.write_text(prof_q, encoding="utf-8")

    # Difference stats
    classical_tokens = encode(classical_out)
    quantum_tokens = encode(quantum_out)
    ham, agree = hamming_and_agreement(classical_tokens, quantum_tokens)
    jacc = jaccard(classical_tokens, quantum_tokens)
    lev = levenshtein(classical_out, quantum_out)

    tokens_trained = MAX_ITERS * batch_size * block_size
    thr_c = tokens_trained / max(1e-9, classical_time)
    thr_q = tokens_trained / max(1e-9, quantum_time)

    params_c = param_count(classical)
    params_q = param_count(quantum)

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
    lines.append(f"- Classical loss: {classical_loss:.4f} (wall: {classical_time:.1f}s, cpu: {cpu_c:.1f}s, rss_max: {rss_c:.1f} MB)")
    lines.append(f"- Quantum loss: {quantum_loss:.4f} (wall: {quantum_time:.1f}s, cpu: {cpu_q:.1f}s, rss_max: {rss_q:.1f} MB)")
    lines.append(f"- Classical params: {params_c:,}")
    lines.append(f"- Quantum params: {params_q:,}")
    lines.append(f"- Classical throughput: {thr_c:.1f} tokens/s")
    lines.append(f"- Quantum throughput: {thr_q:.1f} tokens/s\n")

    lines.append("## Output Difference Stats\n")
    lines.append(f"- Hamming distance (tokens, first {min(len(classical_tokens), len(quantum_tokens))}): {ham}")
    lines.append(f"- Agreement ratio: {agree:.3f}")
    lines.append(f"- Jaccard overlap (token sets): {jacc:.3f}")
    lines.append(f"- Levenshtein distance (chars): {lev}\n")

    lines.append("## Sample Outputs\n")
    lines.append("### Classical GPT\n")
    lines.append("```\n" + classical_out + "\n```\n")
    lines.append("### Quantum GPT\n")
    lines.append("```\n" + quantum_out + "\n```\n")

    lines.append("## Profiling (Top 25 functions)\n")
    lines.append("### Classical\n")
    lines.append("```\n" + PROFILE_CLASSICAL.read_text(encoding="utf-8") + "\n```\n")
    lines.append("### Quantum\n")
    lines.append("```\n" + PROFILE_QUANTUM.read_text(encoding="utf-8") + "\n```\n")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved classical to {classical_path}")
    print(f"Saved quantum to {quantum_path}")
    print(f"Wrote report to {RESULTS_MD}")
    print(f"Profiles: {PROFILE_CLASSICAL}, {PROFILE_QUANTUM}")


if __name__ == "__main__":
    main() 