
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator as Estimator, StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp

# GPT hyperparameters (simplified for quantum integration)
batch_size = 4  # Reduced for quantum simulation
block_size = 8  # Reduced for quantum efficiency
n_embd = 16     # Reduced embedding dimension
n_head = 2      # Reduced number of heads
n_layer = 2     # Reduced number of layers
n_qubits = 4    # Number of qubits for quantum attention
dropout = 0.1
learning_rate = 1e-3

class QuantumAttentionHead(nn.Module):
    """ Quantum-enhanced attention head using Qiskit """

    def __init__(self, head_size, n_qubits=4):
        super().__init__()
        self.head_size = head_size
        self.n_qubits = n_qubits

        # Classical Q, K, V projections 
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Create quantum attention circuit
        self.quantum_attention_circuit = self._build_quantum_attention_circuit()

        # Setup quantum neural network for attention computation
        self.quantum_attention_network = self._setup_quantum_network()

        # Classical attention fallback
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for quantum attention
        self.quantum_scale = nn.Parameter(torch.tensor(0.1))

    def _build_quantum_attention_circuit(self):
        """Build parameterized quantum circuit for attention computation"""
        # Feature map for encoding classical data
        feature_map = ZZFeatureMap(self.n_qubits, reps=1, entanglement='circular')

        # Ansatz for trainable parameters
        ansatz = RealAmplitudes(self.n_qubits, reps=2, entanglement='circular')

        # Combine feature map and ansatz
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        return qc

    def _setup_quantum_network(self):
        """Setup quantum neural network using EstimatorQNN"""
        # Observable for measuring quantum attention weights
        observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1.0)])

        # Create EstimatorQNN
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=self.quantum_attention_circuit,
            observables=observable,
            input_params=self.quantum_attention_circuit.parameters[:self.n_qubits],
            weight_params=self.quantum_attention_circuit.parameters[self.n_qubits:],
            estimator=estimator
        )

        # Wrap in TorchConnector for PyTorch integration
        initial_weights = 0.1 * torch.randn(len(qnn.weight_params))
        return TorchConnector(qnn, initial_weights=initial_weights)

    def forward(self, x):
        B, T, C = x.shape

        # Classical Q, K, V computation
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size) 
        v = self.value(x) # (B,T,head_size)

        # Classical attention computation
        wei_classical = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, T)
        wei_classical = wei_classical.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei_classical = F.softmax(wei_classical, dim=-1)

        # Quantum attention enhancement (simplified approach)
        try:
            # Prepare quantum features by averaging across embedding dimension
            quantum_features = torch.mean(q, dim=-1)  # (B, T)

            # Apply quantum network to each sequence position
            quantum_weights = []
            for b in range(B):
                for t in range(min(T, self.n_qubits)):  # Limit to n_qubits
                    # Normalize features for quantum encoding
                    features = quantum_features[b, :self.n_qubits]
                    features = torch.tanh(features)  # Ensure features are in [-1, 1]

                    # Apply quantum network
                    quantum_output = self.quantum_attention_network(features.unsqueeze(0))
                    quantum_weights.append(quantum_output)

            if quantum_weights:
                quantum_enhancement = torch.stack(quantum_weights).mean()
                quantum_enhancement = torch.sigmoid(quantum_enhancement) * self.quantum_scale
            else:
                quantum_enhancement = 0.0

        except Exception as e:
            print(f"Quantum computation failed, using classical attention: {e}")
            quantum_enhancement = 0.0

        # Combine classical and quantum attention
        wei = wei_classical + quantum_enhancement
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class QuantumMultiHeadAttention(nn.Module):
    """ Multi-head attention with quantum enhancement """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Mix of quantum and classical attention heads
        self.quantum_heads = nn.ModuleList([
            QuantumAttentionHead(head_size) for _ in range(min(num_heads, 2))
        ])
        # Classical heads for remaining
        self.classical_heads = nn.ModuleList([
            ClassicalHead(head_size) for _ in range(max(0, num_heads - 2))
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Combine quantum and classical head outputs
        quantum_outputs = [h(x) for h in self.quantum_heads]
        classical_outputs = [h(x) for h in self.classical_heads] 

        out = torch.cat(quantum_outputs + classical_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out

class ClassicalHead(nn.Module):
    """ Classical attention head for comparison """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    """ Classical feedforward network """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ClassicalMultiHeadAttention(nn.Module):
    """Multi-head attention composed purely of classical heads"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([ClassicalHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ClassicalBlock(nn.Module):
    """Transformer block using classical multi-head attention"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = ClassicalMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ClassicalGPT(nn.Module):
    """Standard GPT model without quantum components"""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[ClassicalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class QuantumBlock(nn.Module):
    """ Quantum-enhanced Transformer block """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = QuantumMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class QuantumGPT(nn.Module):
    """ Quantum-enhanced GPT model """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[QuantumBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training setup functions
def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Quantum-enhanced GPT implementation created successfully!")
print(f"Configuration:")
print(f"- Embedding dimension: {n_embd}")
print(f"- Number of attention heads: {n_head}")  
print(f"- Number of qubits: {n_qubits}")
print(f"- Block size: {block_size}")
print(f"- Batch size: {batch_size}")
