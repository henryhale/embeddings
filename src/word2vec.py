"""Skip-gram word2vec with negative sampling (SGNS), written from scratch.

Two embedding tables are learned: one for words acting as the centre of a
window, one for words acting as context. The training signal is a binary task —
"did these two words really co-occur, or did I invent the pair?" — which avoids
the expensive softmax over the whole vocabulary that made the original word2vec
formulation slow.

Sized for a CPU with 2 cores: tiny-shakespeare trains in a few minutes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corpus import PreparedCorpus, Vocab


@dataclass
class TrainConfig:
    dim: int = 128
    epochs: int = 5
    batch_size: int = 4096
    n_negatives: int = 5
    lr: float = 2e-3
    seed: int = 0
    # 2 physical cores. Using all 4 hyperthreads measurably hurts on this chip.
    threads: int = 2


@dataclass
class TrainProgress:
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    loss: float
    elapsed: float

    @property
    def fraction(self) -> float:
        done = (self.epoch - 1) * self.total_batches + self.batch
        return done / max(1, self.total_epochs * self.total_batches)


class SGNS(nn.Module):
    """Skip-gram with negative sampling."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        # Small uniform init for centre vectors; zeros for context is the
        # standard word2vec choice and converges more cleanly than random.
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """Return mean SGNS loss for a batch.

        centers/contexts: (B,)   negatives: (B, K)
        """
        v = self.center(centers)                     # (B, D)
        u_pos = self.context(contexts)               # (B, D)
        u_neg = self.context(negatives)              # (B, K, D)

        # Real pairs should score high...
        pos_score = (v * u_pos).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # ...invented pairs should score low.
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)   # (B, K)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


@dataclass
class TrainResult:
    vectors: np.ndarray          # (V, D) centre embeddings, L2-normalised
    vectors_raw: np.ndarray      # (V, D) centre embeddings, unnormalised
    vocab: Vocab
    losses: list[float] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0


def train(corpus: PreparedCorpus, cfg: TrainConfig | None = None,
          on_progress: Callable[[TrainProgress], None] | None = None,
          progress_every: int = 20) -> TrainResult:
    """Train SGNS on a prepared corpus and return normalised word vectors."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.threads))
    rng = np.random.default_rng(cfg.seed)

    model = SGNS(len(corpus.vocab), cfg.dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_pairs = corpus.n_pairs
    n_batches = max(1, n_pairs // cfg.batch_size)
    neg_probs = corpus.neg_probs
    losses: list[float] = []
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(n_pairs)
        epoch_loss, seen = 0.0, 0

        for b in range(n_batches):
            idx = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            centers = torch.from_numpy(corpus.centers[idx].astype(np.int64))
            contexts = torch.from_numpy(corpus.contexts[idx].astype(np.int64))
            negatives = torch.from_numpy(
                rng.choice(
                    len(corpus.vocab),
                    size=(len(idx), cfg.n_negatives),
                    p=neg_probs,
                ).astype(np.int64)
            )

            loss = model(centers, contexts, negatives)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            seen += 1
            if on_progress and (b % progress_every == 0 or b == n_batches - 1):
                on_progress(TrainProgress(
                    epoch=epoch, total_epochs=cfg.epochs,
                    batch=b + 1, total_batches=n_batches,
                    loss=epoch_loss / max(1, seen),
                    elapsed=time.perf_counter() - start,
                ))

        losses.append(epoch_loss / max(1, seen))

    raw = model.center.weight.detach().numpy().copy()
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / np.maximum(norms, 1e-9)

    return TrainResult(
        vectors=vectors.astype(np.float32),
        vectors_raw=raw.astype(np.float32),
        vocab=corpus.vocab,
        losses=losses,
        config={**cfg.__dict__},
        stats=corpus.stats(),
        seconds=time.perf_counter() - start,
    )


# --- persistence ------------------------------------------------------------

def save(result: TrainResult, model_dir: str | Path) -> Path:
    """Write vectors, vocabulary and metadata to a directory."""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "vectors.npz",
        vectors=result.vectors,
        vectors_raw=result.vectors_raw,
    )
    result.vocab.save(d / "vocab.json")
    (d / "meta.json").write_text(
        json.dumps(
            {
                "config": result.config,
                "stats": result.stats,
                "losses": result.losses,
                "seconds": result.seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return d


@dataclass
class LoadedModel:
    vectors: np.ndarray
    vocab: Vocab
    meta: dict

    def __len__(self) -> int:
        return len(self.vocab)


def load(model_dir: str | Path) -> LoadedModel:
    d = Path(model_dir)
    npz_path = d / "vectors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No trained vectors at {npz_path}. "
            "Run: python train/train_word2vec.py"
        )
    with np.load(npz_path) as z:
        vectors = z["vectors"]
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return LoadedModel(vectors=vectors, vocab=Vocab.load(d / "vocab.json"), meta=meta)
