#!/usr/bin/env python3
"""Train skip-gram word2vec (SGNS) on a text file.

Examples:
    python train/train_word2vec.py                          # tiny-shakespeare, defaults
    python train/train_word2vec.py --epochs 8 --dim 200
    python train/train_word2vec.py --text data/mytext.txt --min-count 3

Expect roughly 3-8 minutes for tiny-shakespeare at default settings on a
2-core CPU. Artifacts land in models/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import corpus as corpus_mod          # noqa: E402
from src import word2vec as w2v               # noqa: E402
from src.search import neighbors              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--text", type=Path, default=ROOT / "data" / "tinyshakespeare.txt",
                    help="path to a UTF-8 text file")
    ap.add_argument("--out", type=Path, default=ROOT / "models",
                    help="output directory for vectors/vocab/meta")
    ap.add_argument("--dim", type=int, default=128, help="embedding dimensions")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--window", type=int, default=5, help="max context window")
    ap.add_argument("--min-count", type=int, default=5,
                    help="drop tokens appearing fewer times than this")
    ap.add_argument("--max-vocab", type=int, default=None)
    ap.add_argument("--subsample", type=float, default=0.0,
                    help="frequent-word subsampling threshold; 0 disables it "
                         "(default). Only worth enabling (try 1e-3) on corpora "
                         "of tens of millions of tokens — on small ones it "
                         "discards most of the training signal")
    ap.add_argument("--negatives", type=int, default=5,
                    help="negative samples per positive pair")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads; 2 physical cores beats 4 hyperthreads here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", nargs="*", default=["love", "king", "night"],
                    help="words to show nearest neighbours for when done")
    args = ap.parse_args()

    if not args.text.exists():
        print(f"No such text file: {args.text}", file=sys.stderr)
        print("Run: python train/fetch_data.py", file=sys.stderr)
        return 1

    print(f"Reading {args.text}")
    text = corpus_mod.load_text(args.text)

    print("Preparing corpus...")
    t0 = time.perf_counter()
    prepared = corpus_mod.prepare(
        text,
        min_count=args.min_count,
        window=args.window,
        max_vocab=args.max_vocab,
        subsample_t=args.subsample,
        seed=args.seed,
    )
    stats = prepared.stats()
    print(f"  vocabulary          {stats['vocab_size']:,}")
    print(f"  tokens (in vocab)   {stats['tokens_raw']:,}")
    if args.subsample > 0:
        kept = stats["tokens_after_subsampling"]
        pct = 100 * kept / max(1, stats["tokens_raw"])
        print(f"  after subsampling   {kept:,} ({pct:.1f}% retained)")
        if pct < 25:
            print(f"  WARNING: subsampling discarded {100 - pct:.0f}% of the corpus. "
                  "On a corpus this size that starves training — "
                  "consider --subsample 0.")
    print(f"  training pairs      {stats['training_pairs']:,}")
    print(f"  prepared in {time.perf_counter() - t0:.1f}s")

    # Rough guide: fewer than ~50 pairs per word and the vectors stay noisy.
    per_word = stats["training_pairs"] / max(1, stats["vocab_size"])
    if per_word < 50:
        print(f"  NOTE: only ~{per_word:.0f} training pairs per word. "
              "Expect noisy vectors — use a longer text or raise --min-count.")

    cfg = w2v.TrainConfig(
        dim=args.dim, epochs=args.epochs, batch_size=args.batch_size,
        n_negatives=args.negatives, lr=args.lr, seed=args.seed, threads=args.threads,
    )

    last_line = [""]

    def report(p: w2v.TrainProgress) -> None:
        bar_w = 24
        filled = int(bar_w * p.fraction)
        line = (
            f"\r  epoch {p.epoch}/{p.total_epochs} "
            f"[{'#' * filled}{'.' * (bar_w - filled)}] "
            f"batch {p.batch}/{p.total_batches}  loss {p.loss:.4f}  "
            f"{p.elapsed:.0f}s"
        )
        sys.stdout.write(line.ljust(len(last_line[0])))
        sys.stdout.flush()
        last_line[0] = line

    print(f"\nTraining SGNS: dim={args.dim} epochs={args.epochs} "
          f"negatives={args.negatives} threads={args.threads}")
    result = w2v.train(prepared, cfg, on_progress=report)
    print(f"\n  done in {result.seconds:.0f}s")
    print("  loss per epoch: " + ", ".join(f"{l:.4f}" for l in result.losses))

    if len(result.losses) > 1 and result.losses[-1] >= result.losses[0]:
        print("  WARNING: loss did not improve. Try more epochs or a lower --lr.")

    out = w2v.save(result, args.out)
    print(f"\nSaved to {out}/  (vectors.npz, vocab.json, meta.json)")

    # A quick sanity check: neighbours should look semantically plausible.
    probes = [w for w in args.probe if w in result.vocab.stoi]
    if probes:
        print("\nNearest neighbours (sanity check):")
        for w in probes:
            nbs = neighbors(result.vectors, result.vocab, w, k=6)
            joined = ", ".join(f"{n.word} ({n.similarity:.2f})" for n in nbs)
            print(f"  {w:>10} -> {joined}")
    else:
        print("\n(None of the probe words are in this vocabulary.)")

    print("\nNext: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
