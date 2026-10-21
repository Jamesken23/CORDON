"""ConvMHSA backbone used by CORDON."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMHSA(nn.Module):
    """Token-sequence encoder: embedding, MHSA, three CNN max-pools, classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc_q = nn.Linear(embedding_dim, embedding_dim)
        self.fc_k = nn.Linear(embedding_dim, embedding_dim)
        self.fc_v = nn.Linear(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2)
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3)
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=4)
        self.fc1 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        h = self.num_heads
        dh = dim // h
        q = self.fc_q(x).view(b, length, h, dh).transpose(1, 2)
        k = self.fc_k(x).view(b, length, h, dh).transpose(1, 2)
        v = self.fc_v(x).view(b, length, h, dh).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, length, dim)
        return self.norm(self.attn_dropout(self.fc(ctx)) + x)

    def encode(self, emb: torch.Tensor):
        x = self.attention(emb).transpose(1, 2)
        pooled = torch.cat(
            [
                F.adaptive_max_pool1d(self.conv1(x), 1),
                F.adaptive_max_pool1d(self.conv2(x), 1),
                F.adaptive_max_pool1d(self.conv3(x), 1),
            ],
            dim=1,
        ).flatten(1)
        hidden = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(hidden)
        return pooled, hidden, logits

    def forward(self, tokens: torch.Tensor, perturbation: torch.Tensor | None = None):
        emb = self.embeddings(tokens)
        if perturbation is not None:
            emb = emb + perturbation
        return self.encode(emb)
