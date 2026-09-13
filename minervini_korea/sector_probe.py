from pathlib import Path
import time
import pandas as pd
from pykrx import stock

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sector_probe_results"
OUT.mkdir(parents=True, exist_ok=True)

dates = ["20161229", "20201230", "20241230", "20260911"]
rows = []
for d in dates:
    for market in ["KOSPI", "KOSDAQ"]:
        print("fetch", d, market, flush=True)
        try:
            x = stock.get_market_sector_classifications(d, market)
            x = x.reset_index()
            x["query_date"] = d
            x["market"] = market
            rows.append(x)
            print("rows", len(x), "columns", list(x.columns), flush=True)
        except Exception as e:
            print("ERROR", d, market, repr(e), flush=True)
        time.sleep(1.0)

if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "sector_probe.csv", index=False)
    print(out.groupby(["query_date", "market"]).size(), flush=True)
    print(out.head(20).to_string(index=False), flush=True)
else:
    raise RuntimeError("No historical sector classifications returned")
