"""
Daily market snapshot fetcher.

Pulls four metrics:
  1. CNN Fear & Greed Index (from CNN's public dataviz endpoint)
  2. CBOE VIX (via yfinance)
  3. % of S&P 500 stocks above 50-day SMA  (computed from constituents)
  4. % of S&P 500 stocks above 200-day SMA (computed from constituents)

Appends a row to data/snapshots.csv. Idempotent: re-running on the same
date overwrites that day's row.
"""

import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

CSV_PATH = Path("data/snapshots.csv")
HEADERS = [
    "date",
    "fear_greed",
    "vix",
    "spx_above_50ma",
    "spx_above_200ma",
    "fetched_at",
]


# ─── Fear & Greed ──────────────────────────────────────────────────────────
def fetch_fear_greed() -> float:
    """CNN's public dataviz endpoint. Returns the current composite score."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return float(data["fear_and_greed"]["score"])


# ─── VIX ────────────────────────────────────────────────────────────────────
def fetch_vix() -> float:
    """Latest VIX close from Yahoo Finance."""
    hist = yf.Ticker("^VIX").history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("VIX history was empty")
    return float(hist["Close"].iloc[-1])


# ─── S&P 500 breadth ───────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """Pull current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # Yahoo uses '-' instead of '.' for dual-class tickers (BRK.B → BRK-B)
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def fetch_sp500_breadth() -> tuple[float | None, float | None]:
    """% of S&P 500 names above their 50-day SMA and 200-day SMA."""
    tickers = get_sp500_tickers()
    # 1y of daily history is plenty for a 200-day SMA.
    data = yf.download(
        tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=True,
        threads=True,
    )

    above_50 = above_200 = 0
    valid_50 = valid_200 = 0

    for t in tickers:
        try:
            closes = data[t]["Close"].dropna()
        except (KeyError, TypeError):
            continue
        if len(closes) < 50:
            continue
        last = float(closes.iloc[-1])

        sma50 = float(closes.tail(50).mean())
        valid_50 += 1
        if last > sma50:
            above_50 += 1

        if len(closes) >= 200:
            sma200 = float(closes.tail(200).mean())
            valid_200 += 1
            if last > sma200:
                above_200 += 1

    pct_50 = 100.0 * above_50 / valid_50 if valid_50 else None
    pct_200 = 100.0 * above_200 / valid_200 if valid_200 else None
    print(
        f"Breadth computed from {valid_50} (50d) / {valid_200} (200d) "
        f"valid tickers out of {len(tickers)}"
    )
    return pct_50, pct_200


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    today = dt.date.today().isoformat()
    fetched_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    fg = vix = pct50 = pct200 = None
    errors: list[str] = []

    for name, fn in [("fear_greed", fetch_fear_greed), ("vix", fetch_vix)]:
        try:
            val = fn()
            if name == "fear_greed":
                fg = val
            else:
                vix = val
            print(f"{name}: {val:.2f}")
        except Exception as e:
            errors.append(f"{name}: {e!r}")

    try:
        pct50, pct200 = fetch_sp500_breadth()
        print(f"spx_above_50ma:  {pct50:.2f}" if pct50 is not None else "spx_above_50ma: n/a")
        print(f"spx_above_200ma: {pct200:.2f}" if pct200 is not None else "spx_above_200ma: n/a")
    except Exception as e:
        errors.append(f"breadth: {e!r}")

    if errors:
        print("Errors during fetch:", *errors, sep="\n  ", file=sys.stderr)

    # Load existing rows, replace today's if present, append, sort, write.
    rows: list[dict] = []
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != today]

    rows.append({
        "date": today,
        "fear_greed":      f"{fg:.2f}"     if fg     is not None else "",
        "vix":             f"{vix:.2f}"    if vix    is not None else "",
        "spx_above_50ma":  f"{pct50:.2f}"  if pct50  is not None else "",
        "spx_above_200ma": f"{pct200:.2f}" if pct200 is not None else "",
        "fetched_at": fetched_at,
    })
    rows.sort(key=lambda r: r["date"])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    # Fail the job only if everything failed (so a partial outage still logs
    # what we got).
    if all(v is None for v in (fg, vix, pct50, pct200)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
