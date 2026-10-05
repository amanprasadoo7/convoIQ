"""Dataset loading. Imports `datasets` lazily so other modules stay light."""


def load_banking77(split: str = "test") -> tuple[list[str], list[str], list[str]]:
    """Return (texts, label_strings, all_label_names) for a Banking77 split."""
    from datasets import load_dataset

    try:
        ds = load_dataset("PolyAI/banking77")
        names = ds["train"].features["label"].names
        d = ds[split]
        return list(d["text"]), [names[i] for i in d["label"]], list(names)
    except Exception as e:  # noqa: BLE001 - deliberate: any load failure must fall through
        print(f"PolyAI/banking77 failed ({type(e).__name__}: {e}); trying mteb/banking77")
        ds = load_dataset("mteb/banking77")
        d = ds[split]
        names = sorted(set(ds["train"]["label_text"]))
        return list(d["text"]), list(d["label_text"]), names
