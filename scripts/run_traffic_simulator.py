#!/usr/bin/env python3
"""
Sequentially POST each row from test_traffic.csv to POST /process (multipart file).

Use after build_drift_dataset.py. Point BACKEND_URL and MOCK_JWT_TOKEN at your deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload CSV texts to /process.")
    parser.add_argument(
        "--csv",
        default="test_traffic.csv",
        help="CSV from build_drift_dataset.py (columns: upload_order, text).",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
        help="FastAPI base URL (no trailing slash).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MOCK_JWT_TOKEN", "mock-jwt-token-for-academic-project"),
        help="Bearer token matching backend MOCK_JWT_TOKEN.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between requests to avoid overloading GPU memory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-request timeout in seconds (long docs are slow).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        print("CSV must contain a 'text' column.", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {args.token}"}
    url = f"{args.backend.rstrip('/')}/process"
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="POST /process"):
        text = str(row["text"])
        order = row["upload_order"] if "upload_order" in df.columns else idx + 1
        name = f"traffic_{int(order)}.txt"
        files = {"file": (name, text.encode("utf-8"), "text/plain; charset=utf-8")}
        try:
            resp = requests.post(url, files=files, headers=headers, timeout=args.timeout)
            if resp.status_code >= 400:
                errors += 1
                print(f"\nHTTP {resp.status_code} order={order}: {resp.text[:500]}", file=sys.stderr)
            resp.raise_for_status()
        except requests.RequestException as exc:
            errors += 1
            print(f"\nRequest failed order={order}: {exc}", file=sys.stderr)
        time.sleep(args.delay)

    if errors:
        print(f"Completed with {errors} error(s).", file=sys.stderr)
        sys.exit(1)
    print("All requests finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
