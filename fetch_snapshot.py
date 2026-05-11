"""
Daily market snapshot fetcher.

Each run:
  1. Reads existing data/snapshots.csv.
  2. Detects any missing trading days in the last 7 calendar days and backfills
     them using vectorized SMA computation over constituent price history.
  3. Fetches today's snapshot (S&P 500, F&G, VIX, breadth).
  4. Writes everything back to data/snapshots.csv.

Idempotent: re-running on the same date overwrites that day's row, so the
two scheduled crons (22:00 and 23:00 UTC) won't produce duplicates.

Metrics:
  - S&P 500 closing level (^GSPC)
  - CNN Fear & Greed Index (current + historical)
  - CBOE VIX (^VIX)
  - % of S&P 500 stocks above 50-day SMA  (point-in-time)
  - % of S&P 500 stocks above 200-day SMA (point-in-time)
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

# How many calendar days back to scan for gaps each run.
GAP_LOOKBACK_DAYS = 7


# ─── Constituent list ──────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """Wikipedia first; pytickersymbols fallback."""
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


# ─── Fear & Greed ──────────────────────────────────────────────────────────
def fetch_fear_greed_current() -> float:
    """Current composite score from CNN."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return float(r.json()["fear_and_greed"]["score"])


def fetch_fear_greed_history() -> pd.Series:
    """Historical F&G values keyed by date (date index, no tz)."""
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


# ─── VIX / SPX ─────────────────────────────────────────────────────────────
def fetch_index_close(symbol: str) -> float:
    """Latest closing value for ^GSPC or ^VIX."""
    hist = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"{symbol} history was empty")
    return float(hist["Close"].iloc[-1])


def fetch_index_history(symbol: str, period: str = "1mo") -> pd.Series:
    """Closing prices for ^GSPC or ^VIX as a date-indexed Series."""
    df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    s = df["Close"].rename(symbol)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.to_datetime(s.index).normalize()
    return s


# ─── Latest-complete-session helper ────────────────────────────────────────
def latest_complete_session() -> dt.date:
    """Date of the most recent fully-completed US trading session.

    On weekdays past 4:30pm ET we treat today as complete. Otherwise walk back
    to the previous weekday. Holidays naturally fall out when we filter on
    actual yfinance data.
    """
    now = dt.datetime.now(ET)
    today = now.date()
    if now.weekday() < 5 and (now.hour > 16 or (now.hour == 16 and now.minute >= 30)):
        return today
    cur = today - dt.timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= dt.timedelta(days=1)
    return cur


# ─── Constituent price download (chunked + retry) ──────────────────────────
def _download_chunk(chunk: list[str]) -> dict[str, pd.Series]:
    last_err = None
    for attempt in range(1, 4):
        try:
            data = yf.download(
                chunk, period="1y", interval="1d",
                group_by="ticker", progress=False,
                auto_adjust=True, threads=True,
            )
            if data is None or data.empty:
                raise RuntimeError("empty DataFrame")
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
    print(f"  chunk gave up: {last_err!r}", file=sys.stderr)
    return {}


def download_all_constituent_closes(tickers: list[str]) -> pd.DataFrame:
    """Wide DataFrame: date index, one column per ticker."""
    chunk_size = 50
    pieces = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"  Downloading chunk {i // chunk_size + 1} "
              f"({i + 1}–{i + len(chunk)} of {len(tickers)})", flush=True)
        closes = _download_chunk(chunk)
        if closes:
            df = pd.DataFrame(closes)
            pieces.append(df)
        time.sleep(2)
    if not pieces:
        raise RuntimeError("All chunks failed")
    combined = pd.concat(pieces, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    combined.index = pd.to_datetime(combined.index).normalize()
    return combined


# ─── Breadth as full date-indexed Series ───────────────────────────────────
def compute_breadth_series(prices: pd.DataFrame) -> pd.DataFrame:
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


# ─── Today-only breadth (used by main path) ───────────────────────────────
def compute_today_breadth(prices: pd.DataFrame, target_date: dt.date) -> tuple:
    """Breadth as of `target_date`. Filters intraday bars."""
    above_50 = above_200 = 0
    valid_50 = valid_200 = 0
    for col in prices.columns:
        closes = prices[col].dropna()
        closes = closes[closes.index.date <= target_date]
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
    return pct_50, pct_200, valid_50, valid_200


# ─── Reading & writing CSV ─────────────────────────────────────────────────
def read_existing_rows() -> dict[str, dict]:
    """Return existing CSV rows keyed by date string."""
    if not CSV_PATH.exists():
        return {}
    rows = {}
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if "sp500" not in r:
                r["sp500"] = ""
            rows[r["date"]] = r
    return rows


def write_rows(rows_by_date: dict[str, dict]) -> None:
    sorted_dates = sorted(rows_by_date.keys())
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for d in sorted_dates:
            row = rows_by_date[d]
            # Normalize: ensure every header present, ordering preserved
            w.writerow({h: row.get(h, "") for h in HEADERS})


# ─── Gap backfill ──────────────────────────────────────────────────────────
def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{float(v):.2f}"


def backfill_recent_gaps(rows_by_date: dict[str, dict], fetched_at: str) -> int:
    """Find missing trading days in the last GAP_LOOKBACK_DAYS calendar days
    and fill them in. Returns the number of newly added rows."""
    today = latest_complete_session()
    window_start = today - dt.timedelta(days=GAP_LOOKBACK_DAYS)

    # What dates DOES yfinance know about in our window? Use ^GSPC as the
    # authoritative trading-day calendar.
    spx_hist = fetch_index_history("^GSPC", period="1mo")
    if spx_hist.empty:
        print("Gap check: couldn't fetch ^GSPC history; skipping gap backfill")
        return 0

    trading_days = [
        ts.date() for ts in spx_hist.index
        if window_start <= ts.date() <= today
    ]
    missing = [d for d in trading_days if d.isoformat() not in rows_by_date]
    if not missing:
        print(f"Gap check: no gaps in last {GAP_LOOKBACK_DAYS} days "
              f"({len(trading_days)} trading days present)")
        return 0

    print(f"Gap check: {len(missing)} missing day(s) — backfilling: "
          f"{', '.join(d.isoformat() for d in missing)}")

    # Pull the supporting data
    tickers = get_sp500_tickers()
    print(f"  S&P 500 constituents: {len(tickers)} tickers")
    prices = download_all_constituent_closes(tickers)
    print(f"  Got {prices.shape[1]} ticker series over {len(prices)} dates")

    breadth = compute_breadth_series(prices)
    vix_hist = fetch_index_history("^VIX", period="1mo")
    try:
        fg_hist = fetch_fear_greed_history()
    except Exception as e:
        print(f"  F&G history fetch failed ({e!r}); breadth-only backfill")
        fg_hist = pd.Series(dtype=float, name="fear_greed")

    added = 0
    for missing_date in missing:
        ts = pd.Timestamp(missing_date)
        # Look up each metric; missing → blank string
        sp500 = spx_hist.get(ts)
        vix = vix_hist.get(ts)
        b50 = breadth["spx_above_50ma"].get(ts)
        b200 = breadth["spx_above_200ma"].get(ts)
        fg = fg_hist.get(ts)

        # If we got NOTHING for this date, skip it (probably a holiday yfinance
        # has but data for has gaps)
        if all(v is None or (isinstance(v, float) and pd.isna(v))
               for v in (sp500, vix, b50, b200, fg)):
            print(f"    {missing_date.isoformat()}: no data available, skipping")
            continue

        rows_by_date[missing_date.isoformat()] = {
            "date": missing_date.isoformat(),
            "sp500": fmt_val(sp500),
            "fear_greed": fmt_val(fg),
            "vix": fmt_val(vix),
            "spx_above_50ma": fmt_val(b50),
            "spx_above_200ma": fmt_val(b200),
            "fetched_at": fetched_at,
        }
        added += 1
        print(f"    {missing_date.isoformat()}: filled "
              f"sp500={fmt_val(sp500)} vix={fmt_val(vix)} "
              f"50ma={fmt_val(b50)} 200ma={fmt_val(b200)} fg={fmt_val(fg)}")
    return added


# ─── Today's breadth fetch (separate from gap backfill) ───────────────────
def fetch_today_breadth() -> tuple:
    """% S&P 500 above 50/200-day SMA, as of the latest complete session."""
    tickers = get_sp500_tickers()
    target = latest_complete_session()
    print(f"Today's breadth: computing as of {target} ({len(tickers)} tickers)")
    prices = download_all_constituent_closes(tickers)
    p50, p200, n50, n200 = compute_today_breadth(prices, target)
    if p50 is not None:
        print(f"  Above 50d: {p50:.2f}% (from {n50} tickers)")
    if p200 is not None:
        print(f"  Above 200d: {p200:.2f}% (from {n200} tickers)")
    if (n50 or 0) < 50:
        raise RuntimeError(
            f"Only {n50} usable tickers — likely Yahoo rate limit. Re-run later."
        )
    return p50, p200


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    today = dt.date.today().isoformat()
    fetched_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ── Step 1: Load existing rows ────────────────────────────────────────
    rows_by_date = read_existing_rows()
    print(f"Loaded {len(rows_by_date)} existing rows from {CSV_PATH}")

    # ── Step 2: Backfill any recent gaps ──────────────────────────────────
    errors: list[str] = []
    gap_added = 0
    try:
        gap_added = backfill_recent_gaps(rows_by_date, fetched_at)
        print(f"Gap backfill complete: {gap_added} row(s) added")
    except Exception as e:
        traceback.print_exc()
        errors.append(f"gap_backfill: {e!r}")

    # ── Step 3: Fetch today's snapshot ────────────────────────────────────
    sp500 = fg = vix = pct50 = pct200 = None

    for name, fn in [
        ("sp500", lambda: fetch_index_close("^GSPC")),
        ("fear_greed", fetch_fear_greed_current),
        ("vix", lambda: fetch_index_close("^VIX")),
    ]:
        try:
            val = fn()
            if name == "sp500":
                sp500 = val
            elif name == "fear_greed":
                fg = val
            else:
                vix = val
            print(f"Today {name}: {val:.2f}")
        except Exception as e:
            traceback.print_exc()
            errors.append(f"{name}: {e!r}")

    try:
        pct50, pct200 = fetch_today_breadth()
    except Exception as e:
        traceback.print_exc()
        errors.append(f"breadth: {e!r}")

    if errors:
        print("\n=== Errors during fetch ===", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    # Today's row (always overwrites)
    rows_by_date[today] = {
        "date": today,
        "sp500": fmt_val(sp500),
        "fear_greed": fmt_val(fg),
        "vix": fmt_val(vix),
        "spx_above_50ma": fmt_val(pct50),
        "spx_above_200ma": fmt_val(pct200),
        "fetched_at": fetched_at,
    }

    write_rows(rows_by_date)
    print(f"\nWrote {CSV_PATH} with {len(rows_by_date)} total rows")

    # Fail only if literally everything failed
    if (
        all(v is None for v in (sp500, fg, vix, pct50, pct200))
        and gap_added == 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
