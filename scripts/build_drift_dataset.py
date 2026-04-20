#!/usr/bin/env python3
"""
Build drift_baseline.csv and test_traffic.csv for the 150-request simulator.

Phase A (rows 1-50): CNN-style news (~500 words, moderate FK).
Phase B (51-100): Billsum legislative text — shifts readability.
Phase C (101-150): arXiv paper bodies — long, academic drift + Wasserstein.

Requires Hugging Face datasets (first run downloads ~several GB). Set HF_TOKEN if needed.
"""

import os
import sys

import pandas as pd
from datasets import load_dataset


def _load_train_slice(name, config, n_rows, column):
    """
    Load the first n_rows examples from split 'train' without streaming.

    HuggingFace ``datasets`` 1.x does not support ``streaming=True`` on
    ``load_dataset`` (it is forwarded into BuilderConfig and raises). Sliced
    splits like ``train[:100]`` limit what gets materialized/cached.
    """
    slice_split = "train[:%d]" % n_rows
    if config is None:
        ds = load_dataset(name, split=slice_split)
    else:
        ds = load_dataset(name, config, split=slice_split)
    # Arrow Dataset: column access returns a Python list of length len(ds).
    return ds[column]


def build_drift_dataset() -> None:
    print("Loading cnn_dailymail (100 train articles)...", file=sys.stderr)
    cnn_articles = _load_train_slice("cnn_dailymail", "3.0.0", 100, "article")

    print("Loading billsum (50 train rows)...", file=sys.stderr)
    bill_texts = _load_train_slice("billsum", None, 50, "text")

    print("Loading scientific_papers arxiv (50 train rows)...", file=sys.stderr)
    arxiv_articles = _load_train_slice("scientific_papers", "arxiv", 50, "article")

    baseline_df = pd.DataFrame({"text": cnn_articles[:50]})
    out_base = os.environ.get("OUT_BASELINE", "drift_baseline.csv")
    baseline_df.to_csv(out_base, index=False)
    print(f"Wrote {out_base} ({len(baseline_df)} rows).", file=sys.stderr)

    live_traffic = []
    live_traffic.extend(cnn_articles[50:100])
    live_traffic.extend(bill_texts)
    live_traffic.extend(arxiv_articles)

    if len(live_traffic) != 150:
        raise RuntimeError(f"expected 150 live rows, got {len(live_traffic)}")

    traffic_df = pd.DataFrame({
        "upload_order": range(1, 151),
        "text": live_traffic,
    })
    out_live = os.environ.get("OUT_TRAFFIC", "test_traffic.csv")
    traffic_df.to_csv(out_live, index=False)
    print(f"Wrote {out_live} ({len(traffic_df)} rows).", file=sys.stderr)


if __name__ == "__main__":
    build_drift_dataset()
