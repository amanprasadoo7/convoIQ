"""Eyeball Banking77 rows and the label distribution. Needs no model and no GPU.

uv run python -m src.inspect --n 20
uv run python -m src.inspect --n 10 --split train --seed 7
uv run python -m src.inspect --stats
uv run python -m src.inspect --n 5 --json
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from loaders.banking77 import load_banking77

TEXT_WIDTH = 70


def _sample(texts: list[str], labels: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    """Return n (text, label) pairs sampled deterministically from the split."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), k=min(n, len(texts)))
    return [(texts[i], labels[i]) for i in idx]


def show_rows(texts: list[str], labels: list[str], n: int, seed: int, as_json: bool) -> None:
    """Print sampled messages beside their true intent label."""
    rows = _sample(texts, labels, n, seed)
    if as_json:
        print(json.dumps([{"text": t, "true": lab} for t, lab in rows], indent=2))
        return

    label_w = max((len(lab) for _, lab in rows), default=4)
    for i, (text, label) in enumerate(rows, start=1):
        flat = " ".join(text.split())
        clipped = flat if len(flat) <= TEXT_WIDTH else flat[: TEXT_WIDTH - 1] + "..."
        print(f"{i:>3}  {label:<{label_w}}  {clipped}")
    distinct = len({lab for _, lab in rows})
    print(f"\n{len(rows)} of {len(texts)} rows, {distinct} distinct label(s) present.")


def show_stats(texts: list[str], labels: list[str]) -> None:
    """Print per-class counts over the whole split, plus the imbalance range."""
    counts = Counter(labels)
    width = max(len(c) for c in counts)
    top = counts.most_common(1)[0][1]
    bottom = min(counts.values())
    for label, count in counts.most_common():
        print(f"{label:<{width}}  {count:>5}  {'#' * max(1, round(40 * count / top))}")
    spread = top / bottom
    note = (
        "Materially imbalanced -- this is where macro F1 diverges from micro F1."
        if spread >= 1.5
        else "Roughly balanced, so macro F1 and micro F1 stay close on this split."
    )
    print(
        f"\n{len(counts)} classes over {len(labels)} rows. "
        f"Largest {top}, smallest {bottom} (~{spread:.1f}x spread). {note}"
    )


def write_csv(
    path: str, texts: list[str], labels: list[str], n: int, seed: int, all_rows: bool
) -> int:
    """Write (label, text) pairs to a CSV so they can be opened in a spreadsheet or editor."""
    rows = list(zip(texts, labels)) if all_rows else _sample(texts, labels, n, seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "text"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Parse arguments, load the requested split, and print or export what was asked for."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n", type=int, default=10, help="how many rows to show")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stats", action="store_true", help="show class counts for the whole split")
    p.add_argument("--json", action="store_true", help="emit rows as JSON")
    p.add_argument("--csv", metavar="PATH", help="write rows to a CSV file instead of printing")
    p.add_argument("--all", action="store_true", help="with --csv, write every row in order")
    args = p.parse_args()

    texts, labels, _ = load_banking77(args.split)

    if args.csv:
        count = write_csv(args.csv, texts, labels, args.n, args.seed, args.all)
        print(f"Wrote {count} {args.split} rows to {args.csv}")
        return

    print(f"Banking77 {args.split}: {len(texts)} rows, {len(set(labels))} labels\n")
    if args.stats:
        show_stats(texts, labels)
    else:
        show_rows(texts, labels, args.n, args.seed, args.json)


if __name__ == "__main__":
    main()
