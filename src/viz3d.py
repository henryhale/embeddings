"""Plotly figures for the embedding space.

The base cloud is drawn once from a fixed projection; searches add overlay
traces on top of it rather than redrawing the layout. Background words are
deliberately recessive (small, low-opacity) so that highlighted words and the
lines between them carry the eye.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .corpus import Vocab
from .reduce import Projection
from .search import Neighbor, SentencePath
from .theme import Theme, apply_layout, diverging_scale, sequential_scale

_AXIS_TITLES = {
    "PCA": ("PC1", "PC2", "PC3"),
    "t-SNE": ("t-SNE 1", "t-SNE 2", "t-SNE 3"),
    "UMAP": ("UMAP 1", "UMAP 2", "UMAP 3"),
}


def _scene(theme: Theme, method: str) -> dict:
    xt, yt, zt = _AXIS_TITLES.get(method, ("dim 1", "dim 2", "dim 3"))
    axis = dict(
        showbackground=False,
        gridcolor=theme.grid,
        zerolinecolor=theme.axis,
        linecolor=theme.axis,
        tickfont=dict(color=theme.muted, size=10),
    )
    # Axis title styling lives under title.font; the old top-level `titlefont`
    # shorthand was removed in Plotly 6.
    title_font = dict(color=theme.muted, size=11)
    return dict(
        xaxis={**axis, "title": dict(text=xt, font=title_font)},
        yaxis={**axis, "title": dict(text=yt, font=title_font)},
        zaxis={**axis, "title": dict(text=zt, font=title_font)},
        aspectmode="cube",
    )


def base_cloud(projection: Projection, vocab: Vocab, theme: Theme,
               color_by: str = "frequency", clusters: np.ndarray | None = None,
               label_top: int = 0, height: int = 620) -> go.Figure:
    """The word cloud in 3D.

    color_by:
      "frequency" — sequential ramp over log word count (magnitude, so one hue)
      "cluster"   — categorical over KMeans labels (identity)
      "none"      — uniform muted points
    """
    coords = projection.coords
    ids = projection.indices
    words = [vocab.itos[i] for i in ids]
    counts = vocab.counts[ids]

    hover = [f"<b>{w}</b><br>{c:,} occurrences" for w, c in zip(words, counts)]
    fig = go.Figure()

    if color_by == "cluster" and clusters is not None:
        # Identity encoding. The hover label repeats the cluster number, so
        # identity never rests on colour alone.
        palette = theme.series
        for ci in sorted(set(int(c) for c in clusters)):
            m = clusters == ci
            fig.add_trace(go.Scatter3d(
                x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
                mode="markers",
                name=f"Cluster {ci + 1}",
                marker=dict(
                    size=4,
                    color=palette[ci % len(palette)],
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=[f"{h}<br>Cluster {ci + 1}" for h in np.array(hover)[m]],
                hoverinfo="text",
            ))
    elif color_by == "frequency":
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            name="Words",
            marker=dict(
                size=4,
                color=np.log1p(counts),
                colorscale=sequential_scale(theme),
                opacity=0.85,
                line=dict(width=0),
                colorbar=dict(
                    title=dict(text="log(count)", font=dict(color=theme.muted, size=11)),
                    tickfont=dict(color=theme.muted, size=10),
                    thickness=10,
                    len=0.5,
                    outlinewidth=0,
                ),
            ),
            text=hover,
            hoverinfo="text",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            name="Words",
            marker=dict(size=4, color=theme.dim, opacity=0.7, line=dict(width=0)),
            text=hover,
            hoverinfo="text",
        ))

    # Selective direct labels: only the most frequent handful, never every point.
    if label_top > 0:
        order = np.argsort(-counts)[:label_top]
        fig.add_trace(go.Scatter3d(
            x=coords[order, 0], y=coords[order, 1], z=coords[order, 2],
            mode="text",
            text=[words[i] for i in order],
            textposition="top center",
            textfont=dict(color=theme.text_secondary, size=10),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(scene=_scene(theme, projection.method))
    apply_layout(fig, theme, height=height)
    return fig


def add_neighbors(fig: go.Figure, projection: Projection, vocab: Vocab,
                  vectors: np.ndarray, query: str, neighbors: list[Neighbor],
                  theme: Theme) -> go.Figure:
    """Draw the query word and a line out to each of its nearest neighbours."""
    if query not in vocab.stoi:
        return fig
    qid = vocab.stoi[query]
    qpos = projection.coord_of(qid)
    if qpos is None:
        qpos = projection.project_new(vectors[qid])

    xs, ys, zs, labels = [], [], [], []
    for nb in neighbors:
        p = projection.coord_of(nb.vocab_id)
        if p is None:
            p = projection.project_new(vectors[nb.vocab_id])
        # Each line is its own segment, separated by None so Plotly lifts the pen.
        xs += [qpos[0], p[0], None]
        ys += [qpos[1], p[1], None]
        zs += [qpos[2], p[2], None]
        labels.append((p, nb))

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=f"Neighbours of “{query}”",
        line=dict(color=theme.highlight, width=2),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter3d(
        x=[p[0] for p, _ in labels],
        y=[p[1] for p, _ in labels],
        z=[p[2] for p, _ in labels],
        mode="markers+text",
        name="Neighbour words",
        marker=dict(size=8, color=theme.highlight,
                    line=dict(width=2, color=theme.surface)),
        text=[nb.word for _, nb in labels],
        textposition="top center",
        textfont=dict(color=theme.text_primary, size=11),
        hovertext=[f"<b>{nb.word}</b><br>cos {nb.similarity:.3f}" for _, nb in labels],
        hoverinfo="text",
    ))

    fig.add_trace(go.Scatter3d(
        x=[qpos[0]], y=[qpos[1]], z=[qpos[2]],
        mode="markers+text",
        name=f"“{query}”",
        marker=dict(size=13, color=theme.series[0], symbol="diamond",
                    line=dict(width=2, color=theme.surface)),
        text=[query],
        textposition="top center",
        textfont=dict(color=theme.text_primary, size=13),
        hovertext=[f"<b>{query}</b> (query)"],
        hoverinfo="text",
    ))
    return fig


def add_sentence_path(fig: go.Figure, projection: Projection, vectors: np.ndarray,
                      path: SentencePath, theme: Theme, slot: int = 0,
                      label: str | None = None) -> go.Figure:
    """Draw a sentence as an ordered walk connecting its word points."""
    if not path.ok:
        return fig

    color = theme.series[slot % len(theme.series)]
    pts = []
    for vid in path.vocab_ids:
        p = projection.coord_of(vid)
        if p is None:
            p = projection.project_new(vectors[vid])
        pts.append(p)
    pts = np.asarray(pts)

    name = label or " ".join(path.words[:6]) + ("…" if len(path.words) > 6 else "")
    order = [f"{i + 1}. {w}" for i, w in enumerate(path.words)]

    # A single word has no path to draw — just mark the point.
    mode = "lines+markers+text" if len(pts) > 1 else "markers+text"
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode=mode,
        name=name,
        line=dict(color=color, width=4),
        marker=dict(size=7, color=color, line=dict(width=2, color=theme.surface)),
        text=path.words,
        textposition="top center",
        textfont=dict(color=theme.text_primary, size=11),
        hovertext=order,
        hoverinfo="text",
    ))

    if len(pts) > 1:
        # Mark where the sentence starts so direction is readable.
        fig.add_trace(go.Scatter3d(
            x=[pts[0, 0]], y=[pts[0, 1]], z=[pts[0, 2]],
            mode="markers",
            name=f"{name} — start",
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(width=2, color=theme.surface)),
            hovertext=[f"start: {path.words[0]}"],
            hoverinfo="text",
            showlegend=False,
        ))
    return fig


def add_analogy(fig: go.Figure, projection: Projection, vocab: Vocab,
                vectors: np.ndarray, a: str, b: str, c: str,
                results: list[Neighbor], theme: Theme) -> go.Figure:
    """Draw a - b + c as two connected legs plus the resulting words."""
    ids = [vocab.stoi[w] for w in (a, b, c) if w in vocab.stoi]
    if len(ids) != 3:
        return fig

    def pos(vid: int) -> np.ndarray:
        p = projection.coord_of(vid)
        return projection.project_new(vectors[vid]) if p is None else p

    pa, pb, pc = (pos(i) for i in ids)
    fig.add_trace(go.Scatter3d(
        x=[pb[0], pa[0], None, pc[0], pa[0]],
        y=[pb[1], pa[1], None, pc[1], pa[1]],
        z=[pb[2], pa[2], None, pc[2], pa[2]],
        mode="lines",
        name=f"{a} − {b} + {c}",
        line=dict(color=theme.series[2], width=3, dash="dot"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[pa[0], pb[0], pc[0]], y=[pa[1], pb[1], pc[1]], z=[pa[2], pb[2], pc[2]],
        mode="markers+text",
        name="Analogy terms",
        marker=dict(size=9, color=theme.series[2],
                    line=dict(width=2, color=theme.surface)),
        text=[a, b, c],
        textposition="top center",
        textfont=dict(color=theme.text_primary, size=11),
        hoverinfo="text",
    ))

    if results:
        rp = np.asarray([pos(r.vocab_id) for r in results])
        fig.add_trace(go.Scatter3d(
            x=rp[:, 0], y=rp[:, 1], z=rp[:, 2],
            mode="markers+text",
            name="Answers",
            marker=dict(size=10, color=theme.highlight, symbol="diamond",
                        line=dict(width=2, color=theme.surface)),
            text=[r.word for r in results],
            textposition="bottom center",
            textfont=dict(color=theme.text_primary, size=11),
            hovertext=[f"<b>{r.word}</b><br>cos {r.similarity:.3f}" for r in results],
            hoverinfo="text",
        ))
    return fig


def similarity_heatmap(matrix: np.ndarray, words: list[str], theme: Theme,
                       height: int = 460) -> go.Figure:
    """Pairwise cosine similarity.

    Diverging, because cosine has a meaningful zero: positive means related,
    negative means opposed, and the neutral midpoint means unrelated.
    """
    fig = go.Figure(go.Heatmap(
        z=matrix, x=words, y=words,
        colorscale=diverging_scale(theme),
        zmid=0, zmin=-1, zmax=1,
        xgap=2, ygap=2,          # surface gap between cells
        hovertemplate="%{y} ↔ %{x}<br>cos %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="cosine", font=dict(color=theme.muted, size=11)),
            tickfont=dict(color=theme.muted, size=10),
            thickness=10, len=0.7, outlinewidth=0,
        ),
    ))
    fig.update_xaxes(tickfont=dict(color=theme.muted, size=11), showgrid=False)
    fig.update_yaxes(tickfont=dict(color=theme.muted, size=11),
                     showgrid=False, autorange="reversed")
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def loss_curve(losses: list[float], theme: Theme, height: int = 260) -> go.Figure:
    """Training loss per epoch — one series, so no legend box is needed."""
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode="lines+markers",
        line=dict(color=theme.series[0], width=2),
        marker=dict(size=8, color=theme.series[0],
                    line=dict(width=2, color=theme.surface)),
        hovertemplate="epoch %{x}<br>loss %{y:.4f}<extra></extra>",
    ))
    title_font = dict(color=theme.muted, size=11)
    fig.update_xaxes(title=dict(text="Epoch", font=title_font),
                     gridcolor=theme.grid, zeroline=False,
                     tickfont=dict(color=theme.muted, size=11), dtick=1)
    fig.update_yaxes(title=dict(text="SGNS loss", font=title_font),
                     gridcolor=theme.grid, zeroline=False,
                     tickfont=dict(color=theme.muted, size=11))
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig
