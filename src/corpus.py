"""Corpus loading, vocabulary building, and skip-gram pair generation.

The expensive part of word2vec on a CPU-only machine is not the matrix maths —
it is generating training pairs in Python. Everything here is vectorised with
numpy so that a ~200k-token corpus produces its ~1M training pairs in well under
a second.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
HAS_LETTER_RE = re.compile(r"[a-z]")

# Frequent-word subsampling threshold (Mikolov et al.). Words more frequent than
# this fraction of the corpus start getting dropped from the token stream.
#
# OFF BY DEFAULT, and that is deliberate. Subsampling is a large-corpus
# technique: on natural text roughly 80% of tokens sit in the frequency band it
# targets, so it routinely discards 90%+ of the corpus. At billions of tokens
# that is free quality. At tiny-shakespeare scale (~200k tokens) it is ruinous —
# measured retention on a Zipfian corpus of that size:
#
#     t=1e-4 ->  1.0% of tokens kept (12k pairs)   <- starves training
#     t=1e-3 ->  3.2%                (39k pairs)
#     t=1e-2 -> 11.1%               (133k pairs)
#     off    ->  100%              (1.2M pairs)   <- what a small corpus needs
#
# Turn it on (1e-3 is the usual value) only for corpora of tens of millions of
# tokens or more. The unigram^0.75 negative-sampling distribution below already
# damps the influence of very frequent words to some degree.
SUBSAMPLE_T = 0.0

# Exponent applied to the unigram distribution when drawing negative samples.
# 0.75 flattens it, so rare words get chosen as negatives more often than their
# raw frequency would allow.
NEG_SAMPLE_POWER = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, keeping intra-word apostrophes.

    Tokens must contain at least one letter. That keeps forms like "covid19"
    and "3d" while dropping bare numerals — every distinct number would
    otherwise become its own vocabulary entry, which is mostly noise in an
    embedding space.
    """
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        t = t.strip("'")
        if t and HAS_LETTER_RE.search(t):
            out.append(t)
    return out


@dataclass
class Vocab:
    """Maps tokens to contiguous ids, ordered by descending frequency."""

    itos: list[str]
    counts: np.ndarray  # int64, parallel to itos

    def __post_init__(self) -> None:
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def freqs(self) -> np.ndarray:
        """Token frequencies as a probability distribution."""
        return self.counts / self.counts.sum()

    def encode(self, tokens: list[str]) -> np.ndarray:
        """Token strings -> ids, silently dropping out-of-vocabulary tokens."""
        return np.fromiter(
            (self.stoi[t] for t in tokens if t in self.stoi), dtype=np.int32
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos, "counts": self.counts.tolist()}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos=d["itos"], counts=np.array(d["counts"], dtype=np.int64))


def build_vocab(tokens: list[str], min_count: int = 5, max_size: int | None = None) -> Vocab:
    """Build a frequency-ordered vocabulary, dropping tokens below min_count."""
    counter = Counter(tokens)
    items = [(w, c) for w, c in counter.most_common() if c >= min_count]
    if not items:
        raise ValueError(
            f"No token appears at least min_count={min_count} times. "
            "Use a longer text or lower min_count."
        )
    if max_size is not None:
        items = items[:max_size]
    itos = [w for w, _ in items]
    counts = np.array([c for _, c in items], dtype=np.int64)
    return Vocab(itos=itos, counts=counts)


def subsample(ids: np.ndarray, vocab: Vocab, t: float = SUBSAMPLE_T,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly drop very frequent tokens (the/and/of) from the token stream.

    Keeps the corpus informative: without this, a large share of training pairs
    involve a handful of function words that carry little meaning.
    """
    rng = rng or np.random.default_rng()
    if t <= 0:
        return ids
    freqs = vocab.freqs
    # P(keep) = sqrt(t/f) + t/f, clipped to 1 for words rarer than the threshold.
    ratio = t / np.maximum(freqs, 1e-12)
    keep_prob = np.minimum(np.sqrt(ratio) + ratio, 1.0)
    return ids[rng.random(len(ids)) < keep_prob[ids]]


def make_pairs(ids: np.ndarray, window: int = 5,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate (center, context) skip-gram pairs.

    Instead of drawing a random window size per centre word — which forces a
    Python loop — each offset d is kept with probability (window-d+1)/window.
    That has the same expected effect as word2vec's dynamic window shrinking
    (nearer words sampled more often) while staying fully vectorised.
    """
    rng = rng or np.random.default_rng()
    if len(ids) < 2:
        return np.empty(0, np.int32), np.empty(0, np.int32)

    centers, contexts = [], []
    for d in range(1, window + 1):
        left, right = ids[:-d], ids[d:]
        keep = rng.random(len(left)) < (window - d + 1) / window
        if not keep.any():
            continue
        l, r = left[keep], right[keep]
        # Both directions: each word predicts the other.
        centers.append(l)
        contexts.append(r)
        centers.append(r)
        contexts.append(l)

    if not centers:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    return (
        np.concatenate(centers).astype(np.int32),
        np.concatenate(contexts).astype(np.int32),
    )


def negative_sampling_table(vocab: Vocab, power: float = NEG_SAMPLE_POWER) -> np.ndarray:
    """Probability vector over the vocabulary for drawing negative samples."""
    p = vocab.counts.astype(np.float64) ** power
    return p / p.sum()


@dataclass
class PreparedCorpus:
    """Everything the training loop needs, derived once from raw text."""

    vocab: Vocab
    centers: np.ndarray
    contexts: np.ndarray
    neg_probs: np.ndarray
    n_tokens_raw: int
    n_tokens_kept: int

    @property
    def n_pairs(self) -> int:
        return len(self.centers)

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.vocab),
            "tokens_raw": self.n_tokens_raw,
            "tokens_after_subsampling": self.n_tokens_kept,
            "training_pairs": self.n_pairs,
        }


def suggest_subsample_t(n_tokens: int) -> float:
    """Subsampling threshold appropriate to corpus size.

    Off below ~10M tokens, where discarding most of the frequent-word stream
    costs more in training signal than it gains in balance.
    """
    return 1e-3 if n_tokens >= 10_000_000 else 0.0


def prepare(text: str, min_count: int = 5, window: int = 5,
            max_vocab: int | None = None, subsample_t: float = SUBSAMPLE_T,
            seed: int = 0) -> PreparedCorpus:
    """Raw text -> vocabulary, training pairs, and negative-sampling table."""
    rng = np.random.default_rng(seed)
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Text contains no usable word tokens.")

    vocab = build_vocab(tokens, min_count=min_count, max_size=max_vocab)
    ids = vocab.encode(tokens)
    kept = subsample(ids, vocab, t=subsample_t, rng=rng)
    centers, contexts = make_pairs(kept, window=window, rng=rng)
    if len(centers) == 0:
        raise ValueError(
            "No training pairs generated — the corpus is too small. "
            "Try a longer text, or lower min_count."
        )

    return PreparedCorpus(
        vocab=vocab,
        centers=centers,
        contexts=contexts,
        neg_probs=negative_sampling_table(vocab),
        n_tokens_raw=len(ids),
        n_tokens_kept=len(kept),
    )


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
