#!/usr/bin/env python3
"""
LOCAL VERSION — run on your laptop (Python 3.9+, datasets 2.x).
Generates drift_baseline.csv and test_traffic.csv for the 150-request simulator.

PURPOSE: Create REAL distributional drift across 3 phases:
  Phase A (rows   1- 50): ag_news          — short journalistic text    (BASELINE)
  Phase B (rows  51-100): DBpedia-14        — encyclopedic formal prose  (MEDIUM DRIFT)
  Phase C (rows 101-150): arxiv abstracts   — academic/scientific text   (STRONG DRIFT)

WHY THESE DATASETS SHOW DRIFT:
  ag_news  → short sentences, common words, news vocabulary
  DBpedia  → longer sentences, formal vocabulary, encyclopedic structure
  arxiv    → highly technical, domain jargon, complex sentence structure
  A Wasserstein / MMD / readability-based drift detector will clearly flag
  the B→C transitions.

Install:
    pip install "datasets>=2.14.0" pandas tqdm

Run:
    python3 build_drift_dataset_local.py

Push to server:
    scp drift_baseline.csv test_traffic.csv b22bb016@cn06:/path/to/scripts/
"""

import os
import sys

import pandas as pd
from tqdm import tqdm


def _stream_texts(name, config=None, column="text", n_rows=100, split="train"):
    """
    Stream exactly n_rows non-empty texts from a HuggingFace dataset.
    streaming=True — only the rows you need are downloaded, nothing materialized.
    """
    from datasets import load_dataset

    kw = dict(streaming=True, split=split)
    ds = load_dataset(name, config, **kw) if config else load_dataset(name, **kw)

    texts = []
    pbar = tqdm(ds, total=n_rows, desc=name, leave=False)
    for row in pbar:
        val = row.get(column, "") or ""
        val = val.strip()
        if val:
            texts.append(val)
        if len(texts) >= n_rows:
            break
    pbar.close()

    if len(texts) < n_rows:
        raise RuntimeError(
            f"Only got {len(texts)}/{n_rows} rows from '{name}' "
            f"(config={config}, column={column})"
        )
    return texts[:n_rows]


def _load_phase(label, candidates, n_rows, fallback_offset):
    """
    Try each (name, config, column) candidate in order.
    If all fail, fall back to ag_news at fallback_offset.

    candidates: list of (name, config, column) tuples, tried in order.
    """
    for name, config, column in candidates:
        try:
            print(f"Loading {label}: trying {name} [{config or 'default'}]...",
                  file=sys.stderr)
            texts = _stream_texts(name, config=config, column=column, n_rows=n_rows)
            print(f"  OK — {len(texts)} rows from '{name}'.", file=sys.stderr)
            return texts
        except Exception as e:
            print(f"  FAILED ({e})", file=sys.stderr)

    print(f"  All candidates failed. Using ag_news offset {fallback_offset}.",
          file=sys.stderr)
    ag = _stream_texts("ag_news", column="text",
                       n_rows=fallback_offset + n_rows)
    return ag[fallback_offset:fallback_offset + n_rows]


def build_drift_dataset():

    # ── Phase A: ag_news ── journalistic, short, common vocabulary ────────────
    print("\n[Phase A] ag_news — baseline text", file=sys.stderr)
    ag_news_100 = _stream_texts("ag_news", column="text", n_rows=100)
    baseline_texts = ag_news_100[:50]   # → drift_baseline.csv
    phase_a_live   = ag_news_100[50:]   # → test_traffic rows 1-50

    # ── Phase B: encyclopedic formal prose ────────────────────────────────────
    # DBpedia-14: Wikipedia-style encyclopedia entries.
    # Clearly different from news: formal register, longer sentences,
    # structured descriptions, entity-centric vocabulary.
    # fancyzhx/dbpedia_14 is a parquet-native mirror of the original.
    print("\n[Phase B] Encyclopedic text (DBpedia) — medium drift", file=sys.stderr)
    phase_b_raw = _load_phase(
        label="Phase B (encyclopedic)",
        candidates=[
            ("fancyzhx/dbpedia_14", None,       "content"),   # parquet mirror
            ("dbpedia_14",          None,        "content"),   # original
            ("SetFit/dbpedia_14",   None,        "text"),      # another mirror
        ],
        n_rows=50,
        fallback_offset=100,
    )
    # Trim to ~400 chars — encyclopedia entries can be very long
    phase_b = [t[:400].strip() for t in phase_b_raw]

    # ── Phase C: academic/scientific text ─────────────────────────────────────
    # arXiv abstracts: technical jargon, complex sentence structure,
    # domain-specific vocabulary. Maximally drifted from news.
    # ccdv/arxiv-summarization is parquet-native (abstract column).
    # ── Phase C: academic/scientific text ─────────────────────────────────────
    print("\n[Phase C] Academic/scientific text (arXiv) — strong drift", file=sys.stderr)
    phase_c_raw = _load_phase(
        label="Phase C (academic)",
        candidates=[
            ("scientific_papers", "arxiv", "article"),
            ("ccdv/arxiv-summarization", "document", "article"),
        ],
        n_rows=50,
        fallback_offset=150,
    )
    
    # Cap at 15,000 characters to ensure massive word counts trigger the Wasserstein drift,
    # without completely overloading your local RAM.
    phase_c = [t[:15000].strip() for t in phase_c_raw]

    # ── Sanity check: print sample from each phase ────────────────────────────
    print("\n── Sample texts (first 80 chars each) ──", file=sys.stderr)
    print(f"Phase A: {baseline_texts[0][:80]!r}", file=sys.stderr)
    print(f"Phase B: {phase_b[0][:80]!r}", file=sys.stderr)
    print(f"Phase C: {phase_c[0][:80]!r}", file=sys.stderr)

    # ── Write drift_baseline.csv ──────────────────────────────────────────────
    baseline_df = pd.DataFrame({"text": baseline_texts})
    out_base = os.environ.get("OUT_BASELINE", "drift_baseline.csv")
    baseline_df.to_csv(out_base, index=False)
    print(f"\nWrote {out_base} ({len(baseline_df)} rows).", file=sys.stderr)

    # ── Write test_traffic.csv ────────────────────────────────────────────────
    live_traffic = phase_a_live + phase_b + phase_c

    if len(live_traffic) != 150:
        raise RuntimeError(f"Expected 150 live rows, got {len(live_traffic)}")

    traffic_df = pd.DataFrame({
        "upload_order": range(1, 151),
        "phase":        ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "text":         live_traffic,
    })
    out_live = os.environ.get("OUT_TRAFFIC", "test_traffic.csv")
    traffic_df.to_csv(out_live, index=False)
    print(f"Wrote {out_live} ({len(traffic_df)} rows).", file=sys.stderr)

    print("\n── Done. Push to server with:", file=sys.stderr)
    print(f"  scp {out_base} {out_live} b22bb016@cn06:/path/to/scripts/\n",
          file=sys.stderr)


if __name__ == "__main__":
    build_drift_dataset()