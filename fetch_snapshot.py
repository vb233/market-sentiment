"""
Daily market snapshot fetcher.

Pulls five metrics:
  1. S&P 500 closing level (^GSPC, via yfinance)
  2. CNN Fear & Greed Index (CNN's public dataviz endpoint)
  3. CBOE VIX (^VIX, via yfinance)
  4. % of S&P 500 stocks above 50-day SMA  (computed from constituents)
  5. % of S&P 500 stocks above 200-day SMA (computed from constituents)

Breadth is computed as of the *last fully-completed* US trading session,
so values match the EOD numbers Barchart / StockCharts publish for
$S5FI / $S5TH — not a partial intraday bar.

Appends a row to data/snapshots.csv. Idempotent: re-running on the same
date overwrites that day's row.
"""

import csv
import datetime as dt
import io
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

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

ET = ZoneInfo("America/New_York")


# ─── Fear & Greed ──────────────────────────────────────────────────────────
def fetch_fear_greed() -> float:
    """CNN's public dataviz endpoint. Returns the current composite score."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return float(r.json()["fear_and_greed"]["score"])


# ─── VIX ────────────────────────────────────────────────────────────────────
def fetch_vix() -> float:
    """Latest VIX close from Yahoo Finance."""
    hist = yf.Ticker("^VIX").history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("VIX history was empty")
    return float(hist["Close"].iloc[-1])


# ─── S&P 500 close ──────────────────────────────────────────────────────────
def fetch_sp500_close() -> float:
    """Latest S&P 500 close from Yahoo Finance."""
    hist = yf.Ticker("^GSPC").history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("^GSPC history was empty")
    return float(hist["Close"].iloc[-1])


# ─── S&P 500 breadth ───────────────────────────────────────────────────────
def latest_complete_session() -> dt.date:
    """Return the date of the most recently completed US trading session.

    On weekdays past 4:30 PM ET we treat *today* as complete (giving 30 min
    of slack after the 4:00 PM close). Otherwise we walk back to the prior
    weekday. yfinance has no data for market holidays, so any holiday in
    that window naturally falls out when we filter.
    """
    now = dt.datetime.now(ET)
    today = now.date()
    if now.weekday() < 5 and (now.hour > 16 or (now.hour == 16 and now.minute >= 30)):
        return today
    cur = today - dt.timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= dt.timedelta(days=1)
    return cur


def get_sp500_tickers() -> list[str]:
    """Pull current S&P 500 constituents from Wikipedia (with a real UA)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    df = tables[0]
    # Yahoo uses '-' instead of '.' for dual-class tickers (BRK.B → BRK-B)
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def _download_chunk(chunk: list[str]) -> dict[str, pd.Series]:
    """Download Closes for a small list of tickers, with up to 3 retries."""
    last_err = None
    for attempt in range(1, 4):
        try:
            data = yf.download(
                chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if data is None or data.empty:
                raise RuntimeError("empty DataFrame returned")
            out: dict[str, pd.Series] = {}
            if len(chunk) == 1:
                t = chunk[0]
                if "Close" in data.columns:
                    out[t] = data["Close"].dropna()
            else:
                for t in chunk:
                    try:
                        out[t] = data[t]["Close"].dropna()
                    except (KeyError, TypeError):
                        continue
            return out
        except Exception as e:
            last_err = e
            print(f"  chunk attempt {attempt} failed: {e!r}", file=sys.stderr)
            time.sleep(3 * attempt)
    print(f"  chunk gave up after 3 tries: {last_err!r}", file=sys.stderr)
    return {}


def fetch_sp500_breadth() -> tuple[float | None, float | None]:
    """% of S&P 500 names above their 50-day SMA and 200-day SMA, evaluated
    at the close of the last fully-completed trading session."""
    tickers = get_sp500_tickers()
    print(f"Fetched {len(tickers)} S&P 500 tickers")

    target_date = latest_complete_session()
    print(f"Computing breadth as of {target_date} (close)")

    all_closes: dict[str, pd.Series] = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(
            f"Downloading chunk {i // chunk_size + 1} "
            f"({i + 1}–{i + len(chunk)} of {len(tickers)})..."
        )
        all_closes.update(_download_chunk(chunk))
        time.sleep(2)  # polite pause to avoid Yahoo rate limits

    above_50 = above_200 = 0
    valid_50 = valid_200 = 0
    sample_lines: list[str] = []

    for t, raw_closes in all_closes.items():
        # Drop any bars *after* the target date (kills partial intraday bars).
        closes = raw_closes[raw_closes.index.date <= target_date]
        if len(closes) < 50:
            continue
        last = float(closes.iloc[-1])

        sma50 = float(closes.tail(50).mean())
        valid_50 += 1
        is_above_50 = last > sma50
        if is_above_50:
            above_50 += 1

        sma200 = None
        is_above_200 = None
        if len(closes) >= 200:
            sma200 = float(closes.tail(200).mean())
            valid_200 += 1
            is_above_200 = last > sma200
            if is_above_200:
                above_200 += 1

        if len(sample_lines) < 5:
            line = (
                f"  {t:6s} close={last:8.2f}  "
                f"sma50={sma50:8.2f} {'↑' if is_above_50 else '↓'}50"
            )
            if sma200 is not None:
                line += f"  sma200={sma200:8.2f} {'↑' if is_above_200 else '↓'}200"
            sample_lines.append(line)

    print("Sample tickers (close vs SMA):")
    for line in sample_lines:
        print(line)

    pct_50 = 100.0 * above_50 / valid_50 if valid_50 else None
    pct_200 = 100.0 * above_200 / valid_200 if valid_200 else None

    if pct_50 is not None:
        print(f"Above 50d:  {above_50}/{valid_50}  →  {pct_50:.2f}%")
    else:
        print("Above 50d: n/a")
    if pct_200 is not None:
        print(f"Above 200d: {above_200}/{valid_200}  →  {pct_200:.2f}%")
    else:
        print("Above 200d: n/a")

    if valid_50 < 50:
        raise RuntimeError(
            f"Only {valid_50} of {len(tickers)} tickers returned usable price data "
            f"— likely Yahoo rate-limiting. Re-run the workflow in a few minutes."
        )

    return pct_50, pct_200


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    today = dt.date.today().isoformat()
    fetched_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    sp500 = fg = vix = pct50 = pct200 = None
    errors: list[str] = []

    for name, fn in [
        ("sp500", fetch_sp500_close),
        ("fear_greed", fetch_fear_greed),
        ("vix", fetch_vix),
    ]:
        try:
            val = fn()
            if name == "sp500":
                sp500 = val
            elif name == "fear_greed":
                fg = val
            else:
                vix = val
            print(f"{name}: {val:.2f}")
        except Exception as e:
            traceback.print_exc()
            errors.append(f"{name}: {e!r}")

    try:
        pct50, pct200 = fetch_sp500_breadth()
    except Exception as e:
        traceback.print_exc()
        errors.append(f"breadth: {e!r}")

    if errors:
        print("\n=== Errors during fetch ===", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    # Load existing rows, replace today's if present, append, sort, write.
    rows: list[dict] = []
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            # Use DictReader; old rows without `sp500` column will get None.
            reader = csv.DictReader(f)
            for r in reader:
                if r["date"] == today:
                    continue
                # Backfill sp500 column for older snapshots if missing.
                if "sp500" not in r:
                    r["sp500"] = ""
                rows.append(r)

    rows.append({
        "date": today,
        "sp500":           f"{sp500:.2f}"  if sp500  is not None else "",
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

    if all(v is None for v in (sp500, fg, vix, pct50, pct200)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
