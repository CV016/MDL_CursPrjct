#!/usr/bin/env python3
"""
Build drift_baseline.csv and test_traffic.csv for the 150-request simulator.

Phase A (rows 1 50): CNN-style news (~500 words, moderate FK).
Phase B (51 100): Billsum legislative text — shifts readability.
Phase C (101 150): arXiv paper bodies — long, academic drift + Wasserstein.

Requires Hugging Face datasets (first run downloads ~several GB). Set HF_TOKEN if needed.
"""

import os
import sys

import pandas as pd
from datasets import load_dataset


def build_drift_dataset() -> None:
    print("Loading cnn_dailymail (100 train articles, streaming)...", file=sys.stderr)
    cnn_stream = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    it_cnn = iter(cnn_stream)
    cnn_articles = [next(it_cnn)["article"] for _ in range(100)]

    print("Loading billsum (50 train rows, streaming)...", file=sys.stderr)
    bill_stream = load_dataset("billsum", split="train", streaming=True)
    it_bill = iter(bill_stream)
    bill_texts = [next(it_bill)["text"] for _ in range(50)]

    print("Loading scientific_papers arxiv (50 train rows, streaming)...", file=sys.stderr)
    arx_stream = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
    it_arx = iter(arx_stream)
    arxiv_articles = [next(it_arx)["article"] for _ in range(50)]

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
