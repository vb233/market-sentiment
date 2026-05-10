"""
One-shot backfill of historical market snapshots.

For each trading day in the last N months, computes:
  - S&P 500 close (^GSPC)
  - VIX (^VIX)
  - CNN Fear & Greed (from CNN's historical dataviz endpoint, ~1y of history)
  - % of S&P 500 stocks above their 50-day SMA  (point-in-time)
  - % of S&P 500 stocks above their 200-day SMA (point-in-time)

Writes the result to data/snapshots.csv, REPLACING whatever was there. Run
this once via the backfill GitHub Actions workflow, then let the daily
workflow take over.

Note on survivorship bias: we use the CURRENT S&P 500 constituent list
across the entire historical window. Index changes happen ~quarterly, so
for a 12-month backfill the bias is small (typically <1 percentage point
on the breadth values). The numbers should match published $S5FI/$S5TH
within tight tolerance for recent dates.
"""

import csv
import datetime as dt
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# Number of months of history to backfill.
BACKFILL_MONTHS = 12

CSV_PATH = Path("data/snapshots.csv")
HEADERS = [
    "date",
    "sp500",
    "fear_greed",
    "vix",
    "spx_above_50ma",
    "spx_above_200ma",
    "fetched_at",
]

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# ─── Constituents ──────────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """Get S&P 500 constituents. Wikipedia first; pytickersymbols fallback."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e!r}); using pytickersymbols")
        from pytickersymbols import PyTickerSymbols
        pts = PyTickerSymbols()
        stocks = list(pts.get_stocks_by_index("S&P 500"))
        return [s["symbol"].replace(".", "-") for s in stocks if s.get("symbol")]


# ─── Bulk historical price download ────────────────────────────────────────
def download_constituent_history(tickers: list[str], period: str) -> pd.DataFrame:
    """Download daily closes for all tickers. Returns a wide DataFrame
    indexed by date with one column per ticker."""
    chunk_size = 50
    pieces = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(
            f"  Chunk {i // chunk_size + 1}: tickers "
            f"{i + 1}–{i + len(chunk)} of {len(tickers)}",
            flush=True,
        )
        for attempt in range(3):
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                )
                if data is None or data.empty:
                    raise RuntimeError("empty DataFrame")
                if len(chunk) == 1:
                    closes = data[["Close"]].rename(columns={"Close": chunk[0]})
                else:
                    parts = {}
                    for t in chunk:
                        try:
                            parts[t] = data[t]["Close"]
                        except (KeyError, TypeError):
                            continue
                    closes = pd.DataFrame(parts)
                pieces.append(closes)
                break
            except Exception as e:
                print(f"    attempt {attempt + 1} failed: {e!r}", flush=True)
                time.sleep(3 * (attempt + 1))
        time.sleep(1.5)
    if not pieces:
        raise RuntimeError("All chunks failed — Yahoo may be rate-limiting")
    combined = pd.concat(pieces, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    return combined


# ─── Breadth (vectorized over all dates) ──────────────────────────────────
def compute_breadth_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Given a wide DataFrame of constituent closes, return a DataFrame
    indexed by date with breadth percentages."""
    sma50 = prices.rolling(window=50, min_periods=50).mean()
    sma200 = prices.rolling(window=200, min_periods=200).mean()

    valid50 = sma50.notna() & prices.notna()
    valid200 = sma200.notna() & prices.notna()

    above50 = ((prices > sma50) & valid50).sum(axis=1)
    above200 = ((prices > sma200) & valid200).sum(axis=1)

    n50 = valid50.sum(axis=1)
    n200 = valid200.sum(axis=1)

    return pd.DataFrame({
        "spx_above_50ma": (100.0 * above50 / n50).where(n50 > 0),
        "spx_above_200ma": (100.0 * above200 / n200).where(n200 > 0),
    })


# ─── Index history ─────────────────────────────────────────────────────────
def fetch_index_history(symbol: str, period: str) -> pd.Series:
    df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    s = df["Close"].rename(symbol)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


# ─── Fear & Greed history ──────────────────────────────────────────────────
def fetch_fear_greed_history() -> pd.Series:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data["fear_and_greed_historical"]["data"]
    s = pd.Series(
        {pd.Timestamp(int(row["x"]), unit="ms").normalize(): float(row["y"])
         for row in rows},
        name="fear_greed",
    )
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    today = dt.date.today()
    start = today - dt.timedelta(days=BACKFILL_MONTHS * 31)
    # We need ~200 trading days BEFORE `start` to have valid 200-day SMAs at
    # the start of the window, so download 2y total.
    print(f"Backfilling {start} → {today}")

    print("Fetching S&P 500 constituent list...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers")

    print("Downloading 2y of constituent prices...")
    prices = download_constituent_history(tickers, period="2y")
    print(
        f"  Got {prices.shape[1]} ticker series, {len(prices)} dates "
        f"from {prices.index.min().date()} to {prices.index.max().date()}"
    )

    print("Computing breadth across full history...")
    breadth = compute_breadth_series(prices)

    print("Fetching ^GSPC and ^VIX history...")
    spx = fetch_index_history("^GSPC", "2y")
    vix = fetch_index_history("^VIX", "2y")

    print("Fetching CNN Fear & Greed history...")
    try:
        fg = fetch_fear_greed_history()
        print(
            f"  F&G: {fg.index.min().date()} → {fg.index.max().date()} "
            f"({len(fg)} days)"
        )
    except Exception as e:
        print(f"  F&G fetch failed: {e!r} — leaving column blank")
        fg = pd.Series(dtype=float, name="fear_greed")

    # Combine everything on a common date index
    df = pd.DataFrame({"sp500": spx, "vix": vix})
    df.index = pd.to_datetime(df.index).normalize()
    breadth.index = pd.to_datetime(breadth.index).normalize()
    df = df.join(breadth, how="left").join(fg, how="left")

    # Trim to the requested window
    start_ts = pd.Timestamp(start)
    df = df.loc[df.index >= start_ts].copy()

    # Drop rows with no SPX/VIX data (weekends, holidays)
    df = df[df["sp500"].notna() | df["vix"].notna()]

    print(f"\nFinal: {df.index.min().date()} → {df.index.max().date()} "
          f"({len(df)} rows)")
    print("Last 5 rows:")
    print(df.tail(5).to_string())

    # Write CSV
    fetched_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for ts, row in df.iterrows():
            w.writerow({
                "date": ts.date().isoformat(),
                "sp500": f"{row['sp500']:.2f}" if pd.notna(row["sp500"]) else "",
                "fear_greed": f"{row['fear_greed']:.2f}" if pd.notna(row.get("fear_greed")) else "",
                "vix": f"{row['vix']:.2f}" if pd.notna(row["vix"]) else "",
                "spx_above_50ma": f"{row['spx_above_50ma']:.2f}" if pd.notna(row["spx_above_50ma"]) else "",
                "spx_above_200ma": f"{row['spx_above_200ma']:.2f}" if pd.notna(row["spx_above_200ma"]) else "",
                "fetched_at": fetched_at,
            })

    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
