"""Project high-dimensional word vectors down to 3D for display.

The projection is fitted **once** over a fixed set of words and then held
still. That matters more than it sounds: if the layout were refitted on every
search, the whole cloud would rearrange itself between queries and you would
lose any sense of where things are.

Words outside the fitted set (a rare search term, say) still need coordinates.
PCA can place them exactly via its learned transform. t-SNE and UMAP have no
such transform, so a new point is placed at the similarity-weighted centroid of
its nearest fitted neighbours — an approximation, but a stable one that leaves
every existing point untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

METHODS = ("PCA", "t-SNE", "UMAP")


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def available_methods() -> list[str]:
    """Reducers installed right now. UMAP is optional (it needs numba)."""
    return [m for m in METHODS if m != "UMAP" or _umap_available()]


@dataclass
class Projection:
    """A fitted 3D layout over a fixed set of vocabulary indices."""

    coords: np.ndarray        # (n, 3) laid-out positions
    indices: np.ndarray       # (n,) vocabulary ids, parallel to coords
    method: str
    _fitted: object = None    # sklearn estimator, when it supports transform
    _source: np.ndarray = None  # (n, D) original vectors of the fitted set

    def __post_init__(self) -> None:
        self._pos = {int(v): i for i, v in enumerate(self.indices)}

    def has(self, vocab_id: int) -> bool:
        return int(vocab_id) in self._pos

    def coord_of(self, vocab_id: int) -> np.ndarray | None:
        i = self._pos.get(int(vocab_id))
        return None if i is None else self.coords[i]

    def project_new(self, vec: np.ndarray, k: int = 10) -> np.ndarray:
        """Place a vector that was not part of the fit, without moving anything.

        PCA applies its exact learned transform. Manifold methods fall back to
        a similarity-weighted centroid of the nearest fitted neighbours.
        """
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)

        if self.method == "PCA" and self._fitted is not None:
            return self._fitted.transform(vec)[0]

        sims = self._source @ vec.ravel()
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        w = np.clip(sims[top], 0, None)
        if w.sum() <= 1e-9:
            w = np.ones_like(w)
        return (self.coords[top] * w[:, None]).sum(axis=0) / w.sum()


def fit(vectors: np.ndarray, indices: np.ndarray, method: str = "PCA",
        seed: int = 0, perplexity: float = 30.0) -> Projection:
    """Fit a 3D layout over `vectors[indices]`.

    `vectors` is assumed L2-normalised, so a dot product is a cosine similarity.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    indices = np.asarray(indices, dtype=np.int64)
    subset = vectors[indices].astype(np.float32)
    n = len(subset)
    if n < 3:
        raise ValueError(f"Need at least 3 words to lay out a 3D space, got {n}.")

    if method == "PCA":
        est = PCA(n_components=3, random_state=seed)
        coords = est.fit_transform(subset)
        return Projection(coords, indices, method, _fitted=est, _source=subset)

    if method == "t-SNE":
        # Perplexity must stay below the sample count or sklearn raises.
        p = float(max(5.0, min(perplexity, (n - 1) / 3)))
        est = TSNE(
            n_components=3,
            perplexity=p,
            init="pca",
            random_state=seed,
            max_iter=500,
        )
        coords = est.fit_transform(subset)
        return Projection(coords, indices, method, _fitted=None, _source=subset)

    # UMAP
    import umap

    est = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, max(2, n - 1)),
        metric="cosine",
        random_state=seed,
    )
    coords = est.fit_transform(subset)
    return Projection(
        np.asarray(coords), indices, method, _fitted=None, _source=subset
    )


def cluster(vectors: np.ndarray, indices: np.ndarray, k: int = 8,
            seed: int = 0) -> np.ndarray:
    """KMeans labels over the original high-dimensional vectors.

    Clustering before projection, not after, so the groups reflect real
    embedding structure rather than artefacts of squashing to 3D.
    """
    from sklearn.cluster import KMeans

    subset = vectors[np.asarray(indices, dtype=np.int64)]
    k = int(max(1, min(k, len(subset))))
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(subset)
