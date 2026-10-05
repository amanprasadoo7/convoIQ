"""
Get familiar with Banking77 before modelling.

    uv run python -m src.explore_dataset

Prints sizes, the label list, class balance, text lengths, duplicate overlap
and random examples. Saves results/banking77/labels.txt and class_distribution.png.
"""

import random
import statistics
from collections import Counter
from pathlib import Path

from loaders.banking77 import load_banking77


def main() -> None:
    out = Path("results/banking77")
    out.mkdir(exist_ok=True)

    train_x, train_y, names = load_banking77("train")
    test_x, _, _ = load_banking77("test")

    # 1. Sizes
    print(f"train: {len(train_x)} messages | test: {len(test_x)} messages")
    print(f"number of intent labels: {len(names)}\n")

    # 2. The labels themselves (this is what the classifier chooses between)
    label_list = sorted(names)
    (out / "labels.txt").write_text("\n".join(label_list) + "\n")
    print("All labels (also saved to results/banking77/labels.txt):")
    for i, name in enumerate(label_list, 1):
        print(f"  {i:>2}. {name}")

    # 3. Class balance
    counts = Counter(train_y)
    ranked = counts.most_common()
    print(f"\nTrain examples per label: min={ranked[-1][1]}, max={ranked[0][1]}")
    print("Most common:", ranked[:5])
    print("Least common:", ranked[-5:])

    # 4. Text length (in words)
    lengths = [len(t.split()) for t in train_x]
    print(
        f"\nWords per message: median={statistics.median(lengths)}, "
        f"mean={statistics.mean(lengths):.1f}, max={max(lengths)}"
    )

    # 5. Data hygiene: exact duplicates between train and test
    overlap = set(train_x) & set(test_x)
    print(f"Messages appearing in both train and test: {len(overlap)}")

    # 6. Random examples, so you can read the data with your own eyes
    rng = random.Random(0)
    print("\n10 random training examples:")
    for i in rng.sample(range(len(train_x)), 10):
        print(f"  [{train_y[i]}] {train_x[i]}")

    # 7. Chart of class balance
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels_sorted = [name for name, _ in reversed(ranked)]
        values = [counts[name] for name in labels_sorted]
        fig, ax = plt.subplots(figsize=(8, 16))
        ax.barh(labels_sorted, values)
        ax.set_xlabel("training examples")
        ax.set_title("Banking77: examples per intent (train)")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=120)
        print("\nSaved results/banking77/class_distribution.png")
    except ImportError:
        print("\n(matplotlib not installed: run `uv add --dev matplotlib` for the chart)")


if __name__ == "__main__":
    main()
