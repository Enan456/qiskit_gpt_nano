#!/usr/bin/env python3
import time
import torch
from quantum_gpt_implementation import (
    n_embd, n_head, block_size, QuantumGPT,
    ClassicalHead, QuantumAttentionHead,
)

class ComponentEvaluator:
    def __init__(self, vocab_size=65, batch_size=2, seq_len=8, device="cpu"):
        assert seq_len <= block_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)

    def _deterministic_tokens(self):
        t = torch.arange(self.batch_size * self.seq_len, dtype=torch.long)
        return (t % self.vocab_size).view(self.batch_size, self.seq_len).to(self.device)

    def _deterministic_embeddings(self):
        # shape: (B, T, n_embd) with fixed values (no randomness)
        B, T, C = self.batch_size, self.seq_len, n_embd
        b = torch.arange(B, dtype=torch.float32).view(B, 1, 1)
        t = torch.arange(T, dtype=torch.float32).view(1, T, 1)
        c = torch.arange(C, dtype=torch.float32).view(1, 1, C)
        x = 0.01 * (b + 1) + 0.001 * (t + 1) + 0.0001 * (c + 1)
        return x.to(self.device)

    def compare_attention_heads(self, head_size=None):
        if head_size is None:
            head_size = n_embd // n_head
        x = self._deterministic_embeddings()

        ch = ClassicalHead(head_size).to(self.device)
        qh = QuantumAttentionHead(head_size).to(self.device)

        # Make quantum weights deterministic and light
        if hasattr(qh, "quantum_attention_network") and hasattr(qh.quantum_attention_network, "_weights"):
            qh.quantum_attention_network._weights.data.zero_()
        if hasattr(qh, "quantum_scale"):
            qh.quantum_scale.data.fill_(0.1)

        with torch.no_grad():
            t0 = time.time(); out_c = ch(x); t1 = time.time()
            out_q = qh(x); t2 = time.time()

        return {
            "classical_shape": tuple(out_c.shape),
            "quantum_shape": tuple(out_q.shape),
            "mean_abs_diff": torch.mean(torch.abs(out_c - out_q)).item(),
            "classical_time_s": t1 - t0,
            "quantum_time_s": t2 - t1,
        }

    def evaluate_model(self):
        X = self._deterministic_tokens()
        Y = torch.roll(X, shifts=-1, dims=1)  # next-token targets
        model = QuantumGPT(self.vocab_size).to(self.device)

        # Determinize quantum params and measure with quantum ON
        for m in model.modules():
            if isinstance(m, QuantumAttentionHead):
                if hasattr(m, "quantum_attention_network") and hasattr(m.quantum_attention_network, "_weights"):
                    m.quantum_attention_network._weights.data.zero_()
                if hasattr(m, "quantum_scale"):
                    m.quantum_scale.data.fill_(0.1)

        with torch.no_grad():
            _, loss_q = model(X, Y)
        quantum_loss = float(loss_q.item())

        # Toggle quantum OFF (set scale=0) and re-evaluate
        for m in model.modules():
            if isinstance(m, QuantumAttentionHead) and hasattr(m, "quantum_scale"):
                m.quantum_scale.data.zero_()

        with torch.no_grad():
            _, loss_c = model(X, Y)
        classical_loss = float(loss_c.item())

        return {
            "loss_quantum_on": quantum_loss,
            "loss_quantum_off": classical_loss,
            "loss_delta": quantum_loss - classical_loss,
        }

def main():
    evaluator = ComponentEvaluator(vocab_size=65, batch_size=2, seq_len=min(8, block_size), device="cpu")
    attn_stats = evaluator.compare_attention_heads()
    model_stats = evaluator.evaluate_model()

    print("Attention comparison:", attn_stats)
    print("Model eval (quantum ON vs OFF):", model_stats)

if __name__ == "__main__":
    main()