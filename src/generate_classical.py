#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

# add src to path if needed
sys.path.append(str(Path(__file__).parent))
from quantum_gpt_implementation import ClassicalGPT, block_size

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "input.txt"
CKPT_PATH = Path(__file__).resolve().parent.parent / "models" / "classical_gpt.pth"
DEVICE = torch.device("cpu")
GEN_TOKENS = 200


class ClassicalGenerator:
    def __init__(self, data_path: Path, ckpt_path: Path, device: torch.device):
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.encode, self.decode, self.vocab_size = self._build_vocab()
        self.model = self._load_model()

    def _build_vocab(self):
        if not self.data_path.exists():
            raise FileNotFoundError("data/input.txt not found. Run: make data")
        text = self.data_path.read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        return encode, decode, len(chars)

    def _load_model(self):
        if not self.ckpt_path.exists():
            raise FileNotFoundError("models/classical_gpt.pth not found. Run training first (make train)")
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        model = ClassicalGPT(self.vocab_size).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def generate(self, prompt: str = "", max_new_tokens: int = GEN_TOKENS) -> str:
        device = next(self.model.parameters()).device
        if prompt:
            idx_list = self.encode(prompt)
            context = torch.tensor([idx_list], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        # deterministic greedy decode
        for _ in range(max_new_tokens):
            idx_cond = context[:, -block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            context = torch.cat((context, next_token), dim=1)
        return self.decode(context[0].tolist())


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "First Citizen:"
    generator = ClassicalGenerator(DATA_PATH, CKPT_PATH, DEVICE)
    out = generator.generate(prompt, max_new_tokens=200)
    print(out)


if __name__ == "__main__":
    main() 