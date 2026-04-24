#!/usr/bin/env python3
"""Download the tiny-shakespeare corpus (~1.1 MB) into data/.

Usage:
    python train/fetch_data.py
    python train/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEST = ROOT / "data" / "tinyshakespeare.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download if present")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.dest.exists() and not args.force:
        kb = args.dest.stat().st_size / 1024
        print(f"Already present: {args.dest} ({kb:.0f} KB). Use --force to refetch.")
        return 0

    print(f"Downloading {args.url}")
    try:
        with urllib.request.urlopen(args.url, timeout=60) as r:
            data = r.read()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "\nNo internet? The app also accepts pasted text — just skip this "
            "step and use the 'Paste your own text' option.",
            file=sys.stderr,
        )
        return 1

    args.dest.write_bytes(data)
    print(f"Wrote {args.dest} ({len(data) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
