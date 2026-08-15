# Embedding Explorer

Train word vectors from scratch, then fly through the space they live in.

Skip-gram word2vec with negative sampling (SGNS), implemented directly in
PyTorch — no gensim, no pretrained weights. Trained on Tiny Shakespeare or any
text you paste in, then projected to 3D so you can look at it.

## What it does

- **Trains real word vectors** — SGNS with frequency subsampling and a
  unigram^0.75 negative-sampling distribution, the same recipe as the original
  word2vec, in about 100 lines of readable PyTorch
- **3D embedding space** — interactive Plotly scatter, coloured by word
  frequency or by cluster
- **Nearest-word search** — type a word, see lines drawn out to its closest
  neighbours
- **Sentence paths** — a sentence is drawn as a line through its words, in
  order, so you can watch it travel across the space. Up to three at once, for
  comparison
- **Analogies** — `king − man + woman`, drawn as connected legs
- **Similarity heatmap** — pairwise cosine similarity on a diverging scale

## Quick start

```bash
python train/fetch_data.py            # downloads Tiny Shakespeare (~1.1 MB)
python train/train_word2vec.py        # ~3-8 min on a 2-core CPU
streamlit run app.py
```

You can also skip the CLI entirely: launch the app, paste your own text into
the sidebar, and press **Train now**.

## Training options

```bash
python train/train_word2vec.py --help

python train/train_word2vec.py --epochs 8 --dim 200        # richer vectors
python train/train_word2vec.py --text data/mybook.txt      # your own corpus
python train/train_word2vec.py --min-count 3               # keep rarer words
```

Artifacts land in `models/`: `vectors.npz`, `vocab.json`, `meta.json`.

The script prints nearest neighbours for a few probe words when it finishes.
That is the sanity check — if `king` comes back with plausible company, the
vectors learned something. If the neighbours look random, train for more
epochs or feed it more text.

## Notes on making it work

**Corpus size is the thing that matters.** Word vectors learn from repetition,
so a few paragraphs will produce noise no matter how long you train. Tiny
Shakespeare (~200k tokens) is about the smallest corpus that gives recognisable
results. Under ~2,000 tokens the app will tell you rather than waste your time.

**PCA vs t-SNE.** PCA is instant, preserves global structure, and can place a
new word exactly. t-SNE gives tighter, more convincing local neighbourhoods but
is slower and has no way to place new points — so a searched word that falls
outside the laid-out set is approximated at the weighted centroid of its nearest
neighbours. Either way the layout is fitted **once** and held still; searching
never rearranges the space beneath you.

**Analogies need scale.** `king − man + woman → queen` comes from models trained
on billions of tokens. On a corpus this size, treat a good analogy as a happy
accident rather than a benchmark.

## Layout

```
src/corpus.py     tokenizing, vocabulary, subsampling, vectorised pair generation
src/word2vec.py   the SGNS model, training loop, save/load
src/reduce.py     PCA / t-SNE / optional UMAP down to 3D, with stable re-projection
src/search.py     cosine neighbours, analogies, sentence paths
src/viz3d.py      Plotly figures
src/theme.py      validated chart palette, light and dark
train/            CLI training scripts
app.py            Streamlit UI
```
