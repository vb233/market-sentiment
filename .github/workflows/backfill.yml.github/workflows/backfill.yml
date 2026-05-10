name: Backfill historical snapshots

# Manual trigger only — run this once to seed historical data, then let
# the daily-snapshot workflow take over.
on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  backfill:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run backfill
        run: python backfill.py

      - name: Commit backfilled CSV
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/snapshots.csv
          if git diff --staged --quiet; then
            echo "No changes — nothing to commit."
          else
            git commit -m "Backfill historical snapshots"
            git push
          fi
