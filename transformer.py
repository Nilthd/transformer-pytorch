"""
Transformer — built from scratch in PyTorch
============================================
The Transformer replaced RNNs and LSTMs as the dominant architecture
for sequence modeling. Instead of processing tokens one by one,
it looks at the entire sequence at once using "attention" —
each token gets to ask: which other tokens should I pay attention to?

This file builds the full Transformer block by block:
    1. Positional Encoding   — tells the model where each token sits
    2. Single Attention Head — one "perspective" on the sequence
    3. Multi-Head Attention  — many perspectives combined
    4. Feed-Forward Network  — processes each token independently
    5. Transformer Block     — attention + FFN + residual connections
    6. Full Transformer      — stacked blocks for classification

Usage:
    # Train on your own CSV (classification task):
    python transformer.py --data your_data.csv --target label_column

    # Run demo with synthetic data:
    python transformer.py

Author: Niloofar Tavahoodi
"""

import argparse
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Builds overlapping windows from time-series data.
    Each window becomes one "sentence" fed to the Transformer.

    Example with seq_len=4 and data [a, b, c, d, e, f]:
        sample 0 → [a, b, c, d] → predict e
        sample 1 → [b, c, d, e] → predict f
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.targets  = torch.LongTensor(targets.astype(int))
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]   # (seq_len, n_features)
        y = self.targets[idx + self.seq_len]           # scalar class label
        return x, y


# ── 2. Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Transformers have no built-in sense of order — attention treats all
    positions equally. Positional encoding fixes this by adding a unique
    pattern to each position, so the model can distinguish "first token"
    from "fifth token".

    We use sine and cosine waves at different frequencies.
    Think of it like a clock: different hands move at different speeds,
    and their combined position uniquely identifies every moment in time.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # build a (max_len, d_model) table of position encodings
        PE       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        PE[:, 0::2] = torch.sin(position * div_term)   # even dimensions → sine
        PE[:, 1::2] = torch.cos(position * div_term)   # odd  dimensions → cosine

        # register as a buffer so it moves to GPU with the model but isn't trained
        self.register_buffer("PE", PE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add position info to every token in the sequence
        return x + self.PE[: x.size(1), :]


# ── 3. Single Attention Head ──────────────────────────────────────────────────

class SingleHead(nn.Module):
    """
    One attention head — one "way of looking" at the sequence.

    Each token creates three vectors:
        Q (Query)  — what am I looking for?
        K (Key)    — what do I contain?
        V (Value)  — what information do I carry?

    Attention score between token i and token j:
        score = (Q_i · K_j) / sqrt(d_k)

    Dividing by sqrt(d_k) keeps scores from getting too large,
    which would push softmax into a saturated, near-one-hot region.

    The output is a weighted sum of Values — tokens the query
    found relevant contribute more to the result.
    """

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)   # project to query space
        self.W_k = nn.Linear(d_model, d_k)   # project to key space
        self.W_v = nn.Linear(d_model, d_k)   # project to value space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)   # (batch, seq, d_k)
        K = self.W_k(x)   # (batch, seq, d_k)
        V = self.W_v(x)   # (batch, seq, d_k)

        # compute attention scores: how much should each token attend to each other?
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)   # normalize to sum to 1

        # weighted sum of values — this is what the head "sees"
        output = torch.matmul(weights, V)
        return output


# ── 4. Multi-Head Attention ───────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel — each head learns
    to focus on different aspects of the sequence.

    One head might learn syntax, another might track long-range
    dependencies, another might handle local patterns.
    Their outputs are concatenated and projected back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHead(d_model, d_model // num_heads)
            for _ in range(num_heads)
        ])
        # final projection combines all heads back into d_model dimensions
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run all heads independently, then concatenate along the feature dim
        head_outputs = [head(x) for head in self.heads]
        concat       = torch.cat(head_outputs, dim=-1)
        return self.W_o(concat)


# ── 5. Feed-Forward Network ───────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    A small two-layer network applied independently to each token.

    After attention lets tokens communicate with each other,
    this FFN lets each token "think" about what it learned —
    processing its own representation before passing it along.

    The inner layer is typically 4× wider than d_model (here d_ff=256).
    """

    def __init__(self, d_model: int, d_ff: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))


# ── 6. Transformer Block ──────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One full Transformer layer = attention + feed-forward,
    each wrapped with a residual connection and layer normalization.

    Residual connections (x + sublayer(x)) let gradients flow
    directly through, making deep networks much easier to train.

    Layer norm stabilizes activations so training stays smooth.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff        = FeedForward(d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention sub-layer: communicate across tokens
        x = self.norm1(x + self.attention(x))

        # feed-forward sub-layer: process each token independently
        x = self.norm2(x + self.ff(x))

        return x


# ── 7. Full Transformer ───────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Stacks multiple Transformer blocks for sequence classification.

    Input features are projected to d_model dimensions, positional
    encodings are added, then N Transformer blocks process the sequence.
    Finally, we average across all timesteps and classify.

    Why average pooling at the end?
    Unlike RNNs that naturally produce one output at the final step,
    Transformers output one vector per token. Averaging gives a single
    fixed-size representation of the whole sequence.
    """

    def __init__(
        self,
        input_size:  int,
        d_model:     int,
        num_heads:   int,
        num_layers:  int,
        output_size: int,
    ):
        super().__init__()

        # project raw features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)   # raw features → d_model
        x = self.pos_encoding(x)       # add position information

        for block in self.blocks:      # pass through N Transformer blocks
            x = block(x)

        x = x.mean(dim=1)             # average across all timesteps
        x = self.fc(x)                # final classification
        return x


# ── 8. Data Loading ───────────────────────────────────────────────────────────

def load_csv(path: str, target_col: str, seq_len: int, batch_size: int):
    """
    Loads a CSV for sequence classification.
    Target column should contain class labels (integers or strings).
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values

    # encode string labels to integers if needed
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    n_classes = len(np.unique(y))

    # normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # chronological split — important for time-series
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_val,  X_test, y_val,  y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False)

    train_loader = DataLoader(SequenceDataset(X_train, y_train, seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val,   y_val,   seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X_test,  y_test,  seq_len), batch_size=batch_size, shuffle=False)

    print(f"  Loaded   : {path}")
    print(f"  Features : {feature_cols}")
    print(f"  Target   : {target_col} ({n_classes} classes)")
    print(f"  Split    : {len(X_train)} train / {len(X_val)} val / {len(X_test)} test\n")

    return train_loader, val_loader, test_loader, len(feature_cols), n_classes


def make_synthetic(seq_len: int, batch_size: int):
    """
    Generates a synthetic 3-class classification dataset.
    Three sine-wave patterns at different frequencies represent three classes.
    """
    print("  No CSV provided — running demo with synthetic 3-class data.")
    print("  To use your own data: python transformer.py --data file.csv --target column\n")

    n_per_class = 200
    n_features  = 8

    def make_class(freq, n):
        t = np.linspace(0, 4 * np.pi, n)
        return np.column_stack([
            np.sin(freq * t + i) + np.random.normal(0, 0.1, n)
            for i in range(n_features)
        ]).astype(np.float32)

    X = np.vstack([make_class(1, n_per_class),
                   make_class(2, n_per_class),
                   make_class(3, n_per_class)])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

    # shuffle (ok here since classes are balanced)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split1, split2 = int(0.70 * len(X)), int(0.85 * len(X))

    train_loader = DataLoader(SequenceDataset(X[:split1],        y[:split1],        seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X[split1:split2],  y[split1:split2],  seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X[split2:],        y[split2:],        seq_len), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_features, 3


# ── 9. Train / Evaluate ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x_batch, y_batch in loader:
            if train:
                optimizer.zero_grad()

            output = model(x_batch)
            loss   = criterion(output, y_batch)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds       = output.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer on sequence classification")
    parser.add_argument("--data",        type=str,   default=None,  help="Path to CSV file")
    parser.add_argument("--target",      type=str,   default=None,  help="Target column name (class labels)")
    parser.add_argument("--seq_len",     type=int,   default=10,    help="Sequence length (default: 10)")
    parser.add_argument("--d_model",     type=int,   default=32,    help="Model dimension (default: 32)")
    parser.add_argument("--num_heads",   type=int,   default=4,     help="Attention heads (default: 4)")
    parser.add_argument("--num_layers",  type=int,   default=2,     help="Transformer blocks (default: 2)")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--batch_size",  type=int,   default=32,    help="Batch size (default: 32)")
    parser.add_argument("--lr",          type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Transformer — from scratch in PyTorch")
    print("=" * 55)

    # load data
    if args.data:
        if not args.target:
            raise ValueError("Please also specify --target when using --data")
        train_loader, val_loader, test_loader, n_features, n_classes = load_csv(
            args.data, args.target, args.seq_len, args.batch_size
        )
    else:
        train_loader, val_loader, test_loader, n_features, n_classes = make_synthetic(
            args.seq_len, args.batch_size
        )

    # build model
    model = Transformer(
        input_size  = n_features,
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        output_size = n_classes,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"  Model     : Transformer(d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers})")
    print(f"  Classes   : {n_classes}")
    print(f"  Optimizer : Adam (lr={args.lr})")
    print(f"  Scheduler : CosineAnnealingLR")
    print(f"  Epochs    : {args.epochs}\n")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% "
                  f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%")

    # final test evaluation
    model.load_state_dict(torch.load("best_transformer.pt"))
    test_loss, test_acc = run_epoch(model, test_loader, None, criterion, train=False)

    print(f"\n  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}  |  Test Accuracy: {test_acc:.1f}%")
    print(f"  Model saved   : best_transformer.pt")


if __name__ == "__main__":
    main()
