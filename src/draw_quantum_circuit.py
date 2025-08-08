#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from quantum_gpt_implementation import QuantumAttentionHead

OUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT_DIR / "quantum_attention_circuit.png"
TXT_PATH = OUT_DIR / "quantum_attention_circuit.txt"

def main():
    head = QuantumAttentionHead(head_size=8, n_qubits=4)
    # Save matplotlib figure
    head.draw_circuit(filename=str(PNG_PATH), output="mpl")
    # Also save text diagram
    head.draw_circuit(filename=str(TXT_PATH), output="text")
    print(f"Saved circuit to {PNG_PATH} and {TXT_PATH}")

if __name__ == "__main__":
    main() 