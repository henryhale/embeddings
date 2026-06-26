"""Similarity search, analogies, and sentence paths over trained vectors.

Vectors are stored L2-normalised, so cosine similarity is just a dot product
and the whole vocabulary can be scored with a single matrix-vector multiply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Vocab, tokenize


@dataclass
class Neighbor:
    word: str
    vocab_id: int
    similarity: float


def _topk(sims: np.ndarray, k: int, exclude: set[int]) -> list[int]:
    """Indices of the k highest scores, skipping `exclude`."""
    k_eff = min(len(sims), k + len(exclude))
    cand = np.argpartition(-sims, k_eff - 1)[:k_eff]
    cand = cand[np.argsort(-sims[cand])]
    return [int(i) for i in cand if int(i) not in exclude][:k]


def neighbors(vectors: np.ndarray, vocab: Vocab, word: str,
              k: int = 10) -> list[Neighbor]:
    """The k words closest to `word` by cosine similarity."""
    word = word.strip().lower()
    if word not in vocab.stoi:
        return []
    qid = vocab.stoi[word]
    sims = vectors @ vectors[qid]
    return [
        Neighbor(vocab.itos[i], i, float(sims[i]))
        for i in _topk(sims, k, exclude={qid})
    ]


def neighbors_of_vector(vectors: np.ndarray, vocab: Vocab, vec: np.ndarray,
                        k: int = 10, exclude: set[int] | None = None) -> list[Neighbor]:
    """The k words closest to an arbitrary vector."""
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 1e-9:
        v = v / n
    sims = vectors @ v
    return [
        Neighbor(vocab.itos[i], i, float(sims[i]))
        for i in _topk(sims, k, exclude=exclude or set())
    ]


@dataclass
class AnalogyResult:
    """`a` is to `b` as `c` is to ...?"""

    vector: np.ndarray
    results: list[Neighbor]
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def analogy(vectors: np.ndarray, vocab: Vocab, a: str, b: str, c: str,
            k: int = 5) -> AnalogyResult:
    """Solve a - b + c, e.g. king - man + woman."""
    words = [w.strip().lower() for w in (a, b, c)]
    missing = [w for w in words if w not in vocab.stoi]
    if missing:
        return AnalogyResult(np.zeros(vectors.shape[1], np.float32), [], missing)

    ids = [vocab.stoi[w] for w in words]
    vec = vectors[ids[0]] - vectors[ids[1]] + vectors[ids[2]]
    # The source words are almost always the nearest hits; excluding them is
    # what makes the analogy interesting rather than tautological.
    res = neighbors_of_vector(vectors, vocab, vec, k=k, exclude=set(ids))
    return AnalogyResult(vec, res, [])


@dataclass
class SentencePath:
    """A sentence resolved to an ordered walk through the embedding space."""

    words: list[str]        # in-vocabulary tokens, in order
    vocab_ids: list[int]
    oov: list[str]          # tokens dropped as out-of-vocabulary
    raw_tokens: list[str]

    @property
    def ok(self) -> bool:
        return len(self.vocab_ids) >= 1

    def mean_vector(self, vectors: np.ndarray) -> np.ndarray | None:
        """Crude sentence embedding: the average of its word vectors."""
        if not self.vocab_ids:
            return None
        v = vectors[self.vocab_ids].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v


def sentence_path(sentence: str, vocab: Vocab) -> SentencePath:
    """Tokenize a sentence and map it to vocabulary ids, preserving order.

    Repeated words are kept — a sentence that returns to the same word should
    show as a path that revisits that point.
    """
    raw = tokenize(sentence)
    words, ids, oov = [], [], []
    for t in raw:
        if t in vocab.stoi:
            words.append(t)
            ids.append(vocab.stoi[t])
        else:
            oov.append(t)
    return SentencePath(words=words, vocab_ids=ids, oov=oov, raw_tokens=raw)


def similarity_matrix(vectors: np.ndarray, vocab: Vocab,
                      words: list[str]) -> tuple[np.ndarray, list[str]]:
    """Pairwise cosine similarity for a set of words, skipping unknown ones."""
    known = [w for w in (x.strip().lower() for x in words) if w in vocab.stoi]
    if not known:
        return np.zeros((0, 0), np.float32), []
    ids = [vocab.stoi[w] for w in known]
    sub = vectors[ids]
    return (sub @ sub.T).astype(np.float32), known
