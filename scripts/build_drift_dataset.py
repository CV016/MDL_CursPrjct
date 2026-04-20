#!/usr/bin/env python3
"""
Build drift_baseline.csv and test_traffic.csv for the 150-request simulator.

Phase A (rows 1-50): News text (default: AG News; optional CNN/DailyMail via env).
Phase B (51-100): Billsum legislative text — shifts readability.
Phase C (101-150): arXiv paper bodies — long, academic drift + Wasserstein.

Requires Hugging Face ``datasets``. Set HF_TOKEN if needed.

Environment:
  DRIFT_NEWS_DATASET — ``ag_news`` (default, small/reliable) or ``cnn_dailymail``.
  CNN/DailyMail can fail with a corrupted HF cache (NotADirectoryError on
  .../cnn/stories); delete the listed downloads folder or use ag_news.
"""

import os
import sys

import pandas as pd
from datasets import load_dataset


def _download_mode_force():
    """Return a value load_dataset accepts for force re-download (datasets 1.x)."""
    try:
        from datasets import DownloadMode

        return DownloadMode.FORCE_REDOWNLOAD
    except Exception:
        return "force_redownload"


def _load_train_slice(name, config, n_rows, column):
    """
    Load the first n_rows examples from split 'train' without streaming.

    HuggingFace ``datasets`` 1.x does not support ``streaming=True`` on
    ``load_dataset`` (it is forwarded into BuilderConfig and raises). Sliced
    splits like ``train[:100]`` limit what gets materialized/cached.

    If the Hugging Face downloads cache is corrupted (common symptom:
    NotADirectoryError where a directory is expected), retry once with
    ``force_redownload``.
    """
    slice_split = "train[:%d]" % n_rows

    def _invoke(download_mode=None):
        kw = {"ignore_verifications": True}   # <-- add this
        if download_mode is not None:
            kw["download_mode"] = download_mode
        if config is None:
            return load_dataset(name, split=slice_split, **kw)
        return load_dataset(name, config, split=slice_split, **kw)

    try:
        ds = _invoke()
    except (NotADirectoryError, IsADirectoryError) as err:
        # Partial or corrupted tarball: path exists but is not a directory (or vice versa).
        print(
            "Dataset load failed (cache may be corrupted); forcing re-download: "
            "%s" % err,
            file=sys.stderr,
        )
        try:
            ds = _invoke(_download_mode_force())
        except Exception as err2:
            raise RuntimeError(
                "Could not load %r after force_redownload. Remove the broken path "
                "under ~/.cache/huggingface/datasets/downloads/ (see traceback). "
                "For phase A, default DRIFT_NEWS_DATASET=ag_news avoids CNN/DailyMail."
                % name
            ) from err2

    # Arrow Dataset: column access returns a Python list of length len(ds).
    return ds[column]


def _load_news_phase(n_rows):
    """
    Load n_rows news-like texts for phase A.

    Default ``ag_news`` avoids CNN/DailyMail multi-GB extracts that often break
    on partial downloads (NotADirectoryError under .../cnn/stories).
    """
    choice = (os.environ.get("DRIFT_NEWS_DATASET") or "ag_news").strip().lower()
    if choice in ("cnn", "cnn_dailymail", "dailymail"):
        print("Loading cnn_dailymail (%d train rows)..." % n_rows, file=sys.stderr)
        return _load_train_slice("cnn_dailymail", "3.0.0", n_rows, "article")
    if choice not in ("ag_news", "agnews"):
        raise ValueError(
            "DRIFT_NEWS_DATASET must be ag_news or cnn_dailymail, got %r" % choice
        )
    print("Loading ag_news (%d train rows)..." % n_rows, file=sys.stderr)
    return _load_train_slice("ag_news", None, n_rows, "text")


def build_drift_dataset() -> None:
    news_articles = _load_news_phase(100)

    print("Loading ag_news phase B — offset 100 (50 rows)...", file=sys.stderr)
    news_extended = _load_train_slice("ag_news", None, 150, "text")
    bill_texts = news_extended[100:150]

    print("Loading scientific_papers arxiv (50 train rows)...", file=sys.stderr)
    arxiv_articles = _load_train_slice("scientific_papers", "arxiv", 50, "article")

    baseline_df = pd.DataFrame({"text": news_articles[:50]})
    out_base = os.environ.get("OUT_BASELINE", "drift_baseline.csv")
    baseline_df.to_csv(out_base, index=False)
    print(f"Wrote {out_base} ({len(baseline_df)} rows).", file=sys.stderr)

    live_traffic = []
    live_traffic.extend(news_articles[50:100])
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
