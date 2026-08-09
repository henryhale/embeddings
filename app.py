#!/usr/bin/env python3
"""Embedding Explorer — train word vectors and walk through them in 3D.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import corpus as corpus_mod
from src import reduce as reduce_mod
from src import search as search_mod
from src import theme as theme_mod
from src import viz3d
from src import word2vec as w2v

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
TINY_SHAKESPEARE = DATA_DIR / "tinyshakespeare.txt"

st.set_page_config(page_title="Embedding Explorer", page_icon="🧭", layout="wide")


# --- model plumbing ---------------------------------------------------------

def _model_key(model: w2v.LoadedModel) -> str:
    """Cheap identity for caching: shape + a few sampled values."""
    v = model.vectors
    probe = v[:: max(1, len(v) // 8)].sum()
    return f"{v.shape}-{len(model.vocab)}-{probe:.6f}"


@st.cache_resource(show_spinner=False)
def _fit_projection(_vectors: np.ndarray, method: str, n_words: int,
                    seed: int, cache_key: str) -> reduce_mod.Projection:
    """Fit the 3D layout once and hold it. `_vectors` is not hashed."""
    indices = np.arange(min(n_words, len(_vectors)))
    return reduce_mod.fit(_vectors, indices, method=method, seed=seed)


@st.cache_resource(show_spinner=False)
def _fit_clusters(_vectors: np.ndarray, n_words: int, k: int,
                  seed: int, cache_key: str) -> np.ndarray:
    indices = np.arange(min(n_words, len(_vectors)))
    return reduce_mod.cluster(_vectors, indices, k=k, seed=seed)


def _load_saved() -> w2v.LoadedModel | None:
    try:
        return w2v.load(MODEL_DIR)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.error(f"Could not load saved model: {e}")
        return None


def _train_in_app(text: str, cfg: w2v.TrainConfig, min_count: int,
                  window: int) -> w2v.TrainResult | None:
    """Train with a live progress bar, then persist to models/."""
    try:
        with st.spinner("Preparing corpus..."):
            prepared = corpus_mod.prepare(
                text, min_count=min_count, window=window, seed=cfg.seed
            )
    except ValueError as e:
        st.error(str(e))
        return None

    stats = prepared.stats()
    st.caption(
        f"{stats['vocab_size']:,} words in vocabulary · "
        f"{stats['training_pairs']:,} training pairs"
    )

    bar = st.progress(0.0, text="Training...")
    started = time.perf_counter()

    def report(p: w2v.TrainProgress) -> None:
        eta = ""
        if p.fraction > 0.02:
            total = p.elapsed / p.fraction
            eta = f" · about {max(0, total - p.elapsed):.0f}s left"
        bar.progress(
            min(1.0, p.fraction),
            text=f"Epoch {p.epoch}/{p.total_epochs} · loss {p.loss:.4f}{eta}",
        )

    result = w2v.train(prepared, cfg, on_progress=report)
    bar.progress(1.0, text=f"Trained in {time.perf_counter() - started:.0f}s")
    w2v.save(result, MODEL_DIR)
    return result


# --- sidebar ----------------------------------------------------------------

def sidebar() -> tuple[w2v.LoadedModel | None, dict]:
    st.sidebar.title("🧭 Embedding Explorer")

    st.sidebar.subheader("Corpus")
    sources = ["Saved model"]
    if TINY_SHAKESPEARE.exists():
        sources.append("Tiny Shakespeare")
    sources += ["Paste your own text", "Upload a .txt file"]
    source = st.sidebar.radio("Source", sources, label_visibility="collapsed")

    text: str | None = None
    if source == "Tiny Shakespeare":
        text = corpus_mod.load_text(TINY_SHAKESPEARE)
        st.sidebar.caption(f"{len(text):,} characters")
    elif source == "Paste your own text":
        text = st.sidebar.text_area(
            "Your text", height=160,
            placeholder="Paste at least a few thousand words...",
        )
    elif source == "Upload a .txt file":
        up = st.sidebar.file_uploader("Text file", type=["txt", "md"])
        if up is not None:
            text = up.read().decode("utf-8", errors="replace")
            st.sidebar.caption(f"{len(text):,} characters")
    else:
        if not TINY_SHAKESPEARE.exists():
            st.sidebar.caption("Tip: `python train/fetch_data.py` to add Tiny Shakespeare.")

    model: w2v.LoadedModel | None = None
    if source == "Saved model":
        model = _load_saved()
        if model is None:
            st.sidebar.warning(
                "No trained model yet. Pick a corpus above and train, "
                "or run `python train/train_word2vec.py`."
            )
    else:
        st.sidebar.subheader("Training")
        c1, c2 = st.sidebar.columns(2)
        dim = c1.number_input("Dimensions", 16, 512, 128, step=16)
        epochs = c2.number_input("Epochs", 1, 50, 5)
        window = c1.number_input("Window", 1, 15, 5)
        min_count = c2.number_input("Min count", 1, 100, 5)
        negatives = c1.number_input("Negatives", 1, 20, 5)
        lr = c2.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                              value=2e-3, format_func=lambda v: f"{v:g}")

        words = len(corpus_mod.tokenize(text)) if text else 0
        if text:
            st.sidebar.caption(f"~{words:,} word tokens")
        ready = words >= 2000
        if text and not ready:
            st.sidebar.warning(
                f"Only ~{words:,} tokens. Word vectors need repetition to learn "
                "anything — aim for 2,000+ (ideally 50,000+)."
            )

        if st.sidebar.button("Train now", type="primary", disabled=not ready,
                             use_container_width=True):
            cfg = w2v.TrainConfig(dim=int(dim), epochs=int(epochs),
                                  n_negatives=int(negatives), lr=float(lr))
            res = _train_in_app(text, cfg, int(min_count), int(window))
            if res is not None:
                st.cache_resource.clear()
                st.success(f"Trained in {res.seconds:.0f}s — saved to models/")
                model = _load_saved()
        else:
            model = _load_saved()

    opts: dict = {}
    if model is not None:
        st.sidebar.subheader("View")
        methods = reduce_mod.available_methods()
        opts["method"] = st.sidebar.selectbox(
            "Projection", methods,
            help="PCA is instant and places new words exactly. "
                 "t-SNE shows tighter neighbourhoods but is slower.",
        )
        max_words = len(model.vocab)
        opts["n_words"] = st.sidebar.slider(
            "Words shown", 50, max_words, min(600, max_words), step=50,
            help="The most frequent N words. Fewer points is faster and less cluttered.",
        )
        opts["color_by"] = st.sidebar.selectbox(
            "Colour by", ["frequency", "cluster", "none"],
            help="Frequency is a magnitude, so it uses one hue light-to-dark. "
                 "Clusters are identities, so they use distinct hues.",
        )
        if opts["color_by"] == "cluster":
            opts["k"] = st.sidebar.slider("Clusters", 2, 3, 3,
                                          help="Capped at 3: in a scatter every "
                                               "pair of colours is visible at once, "
                                               "and more hues stop being reliably "
                                               "distinguishable for colourblind readers.")
        opts["label_top"] = st.sidebar.slider("Label top N words", 0, 60, 15)

        if opts["method"] == "t-SNE" and opts["n_words"] > 1500:
            st.sidebar.caption("t-SNE on this many points takes a while on 2 cores.")

    return model, opts


# --- main -------------------------------------------------------------------

def main() -> None:
    th = theme_mod.current()
    model, opts = sidebar()

    st.title("Embedding Explorer")
    if model is None:
        st.info(
            "**No trained vectors yet.**\n\n"
            "Quickest start:\n"
            "```bash\n"
            "python train/fetch_data.py\n"
            "python train/train_word2vec.py\n"
            "```\n"
            "Or pick a corpus in the sidebar and press **Train now**."
        )
        st.stop()

    vectors, vocab = model.vectors, model.vocab
    key = _model_key(model)

    with st.spinner(f"Laying out {opts['n_words']:,} words with {opts['method']}..."):
        proj = _fit_projection(vectors, opts["method"], opts["n_words"], 0, key)
        clusters = None
        if opts["color_by"] == "cluster":
            clusters = _fit_clusters(vectors, opts["n_words"], opts.get("k", 3), 0, key)

    tab_explore, tab_compare, tab_model = st.tabs(["Explore", "Compare", "Model"])

    # --- Explore ---
    with tab_explore:
        controls, canvas = st.columns([1, 2.6], gap="medium")

        with controls:
            st.subheader("Search")
            mode = st.radio(
                "Mode",
                ["Nearest words", "Sentence path", "Analogy"],
                label_visibility="collapsed",
            )

            overlays: list = []
            tables: list = []

            if mode == "Nearest words":
                word = st.text_input("Word", value="", placeholder="e.g. king")
                k = st.slider("Neighbours", 3, 30, 10)
                w = word.strip().lower()
                if w and w not in vocab.stoi:
                    st.warning(
                        f"“{w}” is not in the vocabulary. It may be too rare — "
                        "try lowering **Min count** and retraining."
                    )
                elif w:
                    nbs = search_mod.neighbors(vectors, vocab, w, k=k)
                    overlays.append(("neighbors", w, nbs))
                    tables.append((
                        f"Nearest to “{w}”",
                        pd.DataFrame(
                            [(n.word, round(n.similarity, 4)) for n in nbs],
                            columns=["word", "cosine similarity"],
                        ),
                    ))

            elif mode == "Sentence path":
                st.caption(
                    "Each sentence is drawn as a line through its words, in order."
                )
                raw = st.text_area(
                    "Sentence(s) — one per line", height=110,
                    placeholder="to be or not to be\nwhat light through yonder window breaks",
                )
                lines = [l for l in (raw or "").splitlines() if l.strip()][:3]
                if len(([l for l in (raw or "").splitlines() if l.strip()])) > 3:
                    st.caption("Showing the first 3 lines.")
                for i, line in enumerate(lines):
                    path = search_mod.sentence_path(line, vocab)
                    if not path.ok:
                        st.warning(f"No known words in: “{line[:40]}”")
                        continue
                    if path.oov:
                        st.caption(f"Skipped (not in vocabulary): {', '.join(path.oov)}")
                    overlays.append(("path", i, path))
                    tables.append((
                        f"Path {i + 1}",
                        pd.DataFrame(
                            {"step": range(1, len(path.words) + 1), "word": path.words}
                        ),
                    ))

            else:
                st.caption("a − b + c. Classic: king − man + woman.")
                c1, c2, c3 = st.columns(3)
                a = c1.text_input("a", "king")
                b = c2.text_input("b", "man")
                c = c3.text_input("c", "woman")
                if a and b and c:
                    res = search_mod.analogy(vectors, vocab, a, b, c, k=5)
                    if res.missing:
                        st.warning(f"Not in vocabulary: {', '.join(res.missing)}")
                    else:
                        overlays.append(("analogy", (a.lower(), b.lower(), c.lower()),
                                         res.results))
                        tables.append((
                            "Closest to a − b + c",
                            pd.DataFrame(
                                [(n.word, round(n.similarity, 4)) for n in res.results],
                                columns=["word", "cosine similarity"],
                            ),
                        ))
                        st.info(
                            "Small corpora rarely produce textbook analogies — "
                            "there simply isn't enough data. Treat a plausible "
                            "result as a bonus."
                        )

            for title, df in tables:
                st.markdown(f"**{title}**")
                st.dataframe(df, use_container_width=True, hide_index=True)

        with canvas:
            fig = viz3d.base_cloud(
                proj, vocab, th,
                color_by=opts["color_by"],
                clusters=clusters,
                label_top=opts["label_top"],
            )
            for kind, payload, extra in overlays:
                if kind == "neighbors":
                    fig = viz3d.add_neighbors(fig, proj, vocab, vectors,
                                              payload, extra, th)
                elif kind == "path":
                    fig = viz3d.add_sentence_path(fig, proj, vectors, extra, th,
                                                  slot=payload)
                elif kind == "analogy":
                    a, b, c = payload
                    fig = viz3d.add_analogy(fig, proj, vocab, vectors, a, b, c,
                                            extra, th)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displaylogo": False})
            st.caption(
                f"{opts['method']} projection of {len(proj.indices):,} words · "
                "drag to rotate, scroll to zoom"
            )

    # --- Compare ---
    with tab_compare:
        st.subheader("Pairwise similarity")
        st.caption(
            "Cosine similarity between chosen words. Zero means unrelated, so the "
            "scale diverges from a neutral midpoint."
        )
        default = [w for w in vocab.itos[:200]][:8]
        picked = st.multiselect("Words", options=vocab.itos, default=default,
                                max_selections=25)
        if len(picked) >= 2:
            m, known = search_mod.similarity_matrix(vectors, vocab, picked)
            st.plotly_chart(viz3d.similarity_heatmap(m, known, th),
                            use_container_width=True,
                            config={"displaylogo": False})
            with st.expander("Table view"):
                st.dataframe(
                    pd.DataFrame(m, index=known, columns=known).round(3),
                    use_container_width=True,
                )
        else:
            st.info("Pick at least two words.")

    # --- Model ---
    with tab_model:
        meta = model.meta or {}
        stats = meta.get("stats", {})
        cfg = meta.get("config", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vocabulary", f"{len(vocab):,}")
        c2.metric("Dimensions", vectors.shape[1])
        c3.metric("Training pairs", f"{stats.get('training_pairs', 0):,}")
        c4.metric("Train time", f"{meta.get('seconds', 0):.0f}s")

        losses = meta.get("losses") or []
        if losses:
            st.subheader("Training loss")
            st.plotly_chart(viz3d.loss_curve(losses, th), use_container_width=True,
                            config={"displaylogo": False})
            if len(losses) > 1 and losses[-1] >= losses[0]:
                st.warning("Loss did not fall across epochs — try more epochs "
                           "or a lower learning rate.")

        st.subheader("Most frequent words")
        top = pd.DataFrame({
            "word": vocab.itos[:40],
            "count": vocab.counts[:40],
        })
        st.dataframe(top, use_container_width=True, hide_index=True)

        if cfg:
            with st.expander("Hyperparameters"):
                st.json(cfg)


if __name__ == "__main__":
    main()
