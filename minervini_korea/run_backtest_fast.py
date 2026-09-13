from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as base

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def sell_tax_rate(date, market: str) -> float:
    """Total sell-side transaction levy for KOSPI/KOSDAQ retail backtest.

    KOSPI totals include the 0.15% rural special tax where applicable.
    """
    d = pd.Timestamp(date)
    if market not in {"KOSPI", "KOSDAQ"}:
        return 0.0
    if d < pd.Timestamp("2019-06-03"):
        return 0.0030
    if d < pd.Timestamp("2021-01-01"):
        return 0.0025
    if d < pd.Timestamp("2023-01-01"):
        return 0.0023
    if d < pd.Timestamp("2024-01-01"):
        return 0.0020
    if d < pd.Timestamp("2025-01-01"):
        return 0.0018
    if d < pd.Timestamp("2026-01-01"):
        return 0.0015
    return 0.0020


def build_context(df: pd.DataFrame) -> dict:
    cols = [
        "Date", "Code", "Name", "Market", "adj_open", "adj_high", "adj_low",
        "adj_close", "ma20", "ma60"
    ]
    by_date = {}
    for d, q in df[cols].groupby("Date", sort=True):
        by_date[pd.Timestamp(d)] = q.drop(columns="Date").set_index("Code", drop=False)
    return {
        "dates": list(by_date.keys()),
        "by_date": by_date,
        "last_date": df.groupby("Code", sort=False).Date.max().to_dict(),
    }


def build_signal_book(df: pd.DataFrame, p: base.Params) -> dict:
    mask = base.signal_mask(df, p)
    cols = ["Date", "Code", "Name", "pivot20", "rs", "rsacc20", "breakvol"]
    sig = df.loc[mask, cols]
    if sig.empty:
        return {}
    return {
        pd.Timestamp(d): q.to_dict("records")
        for d, q in sig.groupby("Date", sort=False)
    }


def simulate_fast(df: pd.DataFrame, ctx: dict, p: base.Params):
    dates = ctx["dates"]
    by_date = ctx["by_date"]
    last_date = ctx["last_date"]
    signal_book = build_signal_book(df, p)

    cash = 1.0
    pos = {}
    pending_entries = []
    pending_exits = set()
    trades = []
    curve = []

    for d in dates:
        day = by_date[d]

        # Mark only held names. Suspended names keep the last observed close.
        for c, q in pos.items():
            if c in day.index:
                q["last_close"] = float(day.at[c, "adj_close"])

        # Prior-close trailing exits execute at the next available open.
        executed = []
        for c in list(pending_exits):
            if c not in pos or c not in day.index:
                continue
            row = day.loc[c]
            q = pos[c]
            fill = float(row.adj_open) * (1 - base.SLIP)
            tax = sell_tax_rate(d, q["Market"])
            proceeds = q["shares"] * fill * (1 - base.SELL_COMM - tax)
            cash += proceeds
            trades.append({
                "Code": c,
                "Name": q["Name"],
                "Market": q["Market"],
                "entry": q["entry_date"],
                "exit": d,
                "entry_px": q["entry_px"],
                "exit_px": fill,
                "ret": proceeds / q["cost"] - 1,
                "reason": "trend",
                "hold_days": (d - q["entry_date"]).days,
                "sell_tax": tax,
            })
            del pos[c]
            executed.append(c)
        for c in executed:
            pending_exits.discard(c)

        # Prior-day signals enter at D+1 open; avoid >5% gap chase.
        candidates = []
        for sig in pending_entries:
            c = sig["Code"]
            if c in pos or c not in day.index:
                continue
            row = day.loc[c]
            pivot = float(sig["pivot20"])
            gap = float(row.adj_open) / pivot - 1 if pivot > 0 else 9.0
            if gap > 0.05:
                continue
            score = (
                float(sig["rs"])
                + 0.35 * max(float(sig["rsacc20"]), 0.0)
                + 2.0 * min(float(sig["breakvol"]), 3.0)
            )
            candidates.append((score, c, sig, row))
        candidates.sort(reverse=True, key=lambda z: z[0])

        slots = base.MAX_POS - len(pos)
        if slots > 0 and candidates and cash > 0:
            equity_before = cash + sum(q["shares"] * q["last_close"] for q in pos.values())
            target = equity_before / base.MAX_POS
            for _, c, sig, row in candidates[:slots]:
                alloc = min(target, cash)
                if alloc < equity_before * 0.03:
                    continue
                fill = float(row.adj_open) * (1 + base.SLIP)
                gross_per_share = fill * (1 + base.BUY_COMM)
                shares = alloc / gross_per_share
                cost = shares * gross_per_share
                cash -= cost
                pos[c] = {
                    "shares": shares,
                    "entry_px": fill,
                    "entry_date": d,
                    "cost": cost,
                    "Name": sig["Name"],
                    "Market": str(row.Market),
                    "peak": float(row.adj_high),
                    "last_close": float(row.adj_close),
                }
        pending_entries = []

        # Initial hard stop, then progressively looser winner trailing rules.
        stopped = []
        for c, q in list(pos.items()):
            if c not in day.index:
                continue
            row = day.loc[c]
            q["peak"] = max(q["peak"], float(row.adj_high))
            q["last_close"] = float(row.adj_close)
            peak_gain = q["peak"] / q["entry_px"] - 1

            if peak_gain < 0.20:
                stop_px = q["entry_px"] * (1 - p.stop)
                if float(row.adj_low) <= stop_px:
                    raw_fill = float(row.adj_open) if float(row.adj_open) < stop_px else stop_px
                    fill = raw_fill * (1 - base.SLIP)
                    tax = sell_tax_rate(d, q["Market"])
                    proceeds = q["shares"] * fill * (1 - base.SELL_COMM - tax)
                    cash += proceeds
                    trades.append({
                        "Code": c,
                        "Name": q["Name"],
                        "Market": q["Market"],
                        "entry": q["entry_date"],
                        "exit": d,
                        "entry_px": q["entry_px"],
                        "exit_px": fill,
                        "ret": proceeds / q["cost"] - 1,
                        "reason": "stop",
                        "hold_days": (d - q["entry_date"]).days,
                        "sell_tax": tax,
                    })
                    stopped.append(c)
                    continue

            # A delisted/disappearing name is liquidated on its last available close.
            if d == last_date.get(c):
                fill = float(row.adj_close) * (1 - base.SLIP)
                tax = sell_tax_rate(d, q["Market"])
                proceeds = q["shares"] * fill * (1 - base.SELL_COMM - tax)
                cash += proceeds
                trades.append({
                    "Code": c,
                    "Name": q["Name"],
                    "Market": q["Market"],
                    "entry": q["entry_date"],
                    "exit": d,
                    "entry_px": q["entry_px"],
                    "exit_px": fill,
                    "ret": proceeds / q["cost"] - 1,
                    "reason": "last_row",
                    "hold_days": (d - q["entry_date"]).days,
                    "sell_tax": tax,
                })
                stopped.append(c)
                continue

            if peak_gain >= 0.50:
                trail = row.ma60
            elif peak_gain >= 0.20:
                trail = row.ma20
            else:
                trail = np.nan
            if pd.notna(trail) and float(row.adj_close) < float(trail):
                pending_exits.add(c)

        for c in stopped:
            pos.pop(c, None)
            pending_exits.discard(c)

        equity = cash + sum(q["shares"] * q["last_close"] for q in pos.values())
        curve.append((d, equity, len(pos), cash))
        pending_entries = signal_book.get(d, [])

    # Final liquidation at the last marked close.
    d = dates[-1]
    day = by_date[d]
    for c, q in list(pos.items()):
        px = float(day.at[c, "adj_close"]) if c in day.index else q["last_close"]
        fill = px * (1 - base.SLIP)
        tax = sell_tax_rate(d, q["Market"])
        proceeds = q["shares"] * fill * (1 - base.SELL_COMM - tax)
        cash += proceeds
        trades.append({
            "Code": c,
            "Name": q["Name"],
            "Market": q["Market"],
            "entry": q["entry_date"],
            "exit": d,
            "entry_px": q["entry_px"],
            "exit_px": fill,
            "ret": proceeds / q["cost"] - 1,
            "reason": "end",
            "hold_days": (d - q["entry_date"]).days,
            "sell_tax": tax,
        })
        del pos[c]

    if curve:
        curve[-1] = (curve[-1][0], cash, 0, cash)
    curve = pd.DataFrame(curve, columns=["Date", "equity", "npos", "cash"]).set_index("Date")
    trades = pd.DataFrame(trades)
    return curve, trades


def drawdown_episodes(curve: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    e = curve.equity
    dd = e / e.cummax() - 1
    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough = 0.0
    for d, v in dd.items():
        if v < 0 and not in_dd:
            in_dd = True
            peak_date = e.loc[:d].idxmax()
            trough_date = d
            trough = float(v)
        elif in_dd:
            if v < trough:
                trough = float(v)
                trough_date = d
            if v >= -1e-12:
                episodes.append({
                    "peak": peak_date,
                    "trough": trough_date,
                    "recovery": d,
                    "drawdown": trough,
                    "days_to_trough": (trough_date - peak_date).days,
                    "days_to_recovery": (d - peak_date).days,
                })
                in_dd = False
    if in_dd:
        episodes.append({
            "peak": peak_date,
            "trough": trough_date,
            "recovery": pd.NaT,
            "drawdown": trough,
            "days_to_trough": (trough_date - peak_date).days,
            "days_to_recovery": np.nan,
        })
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame(episodes).sort_values("drawdown").head(topn)


def tests():
    out = [
        base.Params("full_stack_vcp", full_stack=True),
        base.Params("early_vcp_no_accel", rsacc=-100),
        base.Params("early_vcp", rsacc=10),
        base.Params("early_no_vcp", rsacc=10, use_vcp=False),
        base.Params("early_vcp_breadth45", rsacc=10, breadth=0.45),
        base.Params("early_vcp_breadth50", rsacc=10, breadth=0.50),
        base.Params("early_vcp_breadth55", rsacc=10, breadth=0.55),
    ]
    for rs in [85, 90, 95]:
        for ratio in [0.95, 0.97, 0.99]:
            out.append(base.Params(f"grid_rs{rs}_ma{ratio:.2f}", rs=rs, ma_ratio=ratio))
    for stop in [0.05, 0.06, 0.07, 0.08]:
        out.append(base.Params(f"grid_stop{int(stop*100)}", stop=stop))
    for bv in [1.0, 1.2, 1.3, 1.5]:
        out.append(base.Params(f"grid_breakvol{bv:.1f}", breakvol=bv))
    for vr in [0.70, 0.80, 0.90]:
        out.append(base.Params(f"grid_voldry{vr:.2f}", vol_dry=vr))
    return list({p.name: p for p in out}.values())


def main():
    print("Loading indicators...", flush=True)
    df = base.load_prepare()
    print(f"rows={len(df):,} tickers={df.Code.nunique():,}", flush=True)
    ctx = build_context(df)
    print(f"dates={len(ctx['dates']):,}", flush=True)

    manifest = {
        "rows": int(len(df)),
        "tickers": int(df.Code.nunique()),
        "start": str(df.Date.min().date()),
        "end": str(df.Date.max().date()),
        "source": "FinanceData/marcap",
        "execution": "optimized holdings-and-signals-only simulator",
        "tax_schedule": {
            "2015_to_2019-06-02": 0.0030,
            "2019-06-03_to_2020": 0.0025,
            "2021_to_2022": 0.0023,
            "2023": 0.0020,
            "2024": 0.0018,
            "2025": 0.0015,
            "2026": 0.0020,
        },
        "notes": [
            "Signals use D close and entries use D+1 open.",
            "20 bps one-way slippage and 1.5 bps one-way commission.",
            "Pending exits on suspended names wait for the next available row.",
            "Sector and filing-time fundamentals are not included in this price-only phase.",
        ],
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries = []
    curves = {}
    tradebooks = {}
    params = tests()
    for i, p in enumerate(params, 1):
        print(f"[{i}/{len(params)}] {p.name}", flush=True)
        c, t = simulate_fast(df, ctx, p)
        m = base.metrics(c, t)
        m["Strategy"] = p.name
        m.update({
            "rs": p.rs,
            "rsacc": p.rsacc,
            "ma_ratio": p.ma_ratio,
            "last_range": p.last_range,
            "vol_dry": p.vol_dry,
            "atr_ratio": p.atr_ratio,
            "breakvol": p.breakvol,
            "stop": p.stop,
            "breadth": p.breadth,
            "full_stack": p.full_stack,
            "use_vcp": p.use_vcp,
        })
        summaries.append(m)
        curves[p.name] = c
        tradebooks[p.name] = t
        pd.DataFrame(summaries).to_csv(RESULTS / "checkpoint_summary.csv", index=False)

    summary = pd.DataFrame(summaries).sort_values(["Calmar", "CAGR"], ascending=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    eligible = summary[summary.Trades >= 20]
    best = (eligible if len(eligible) else summary).iloc[0]
    bname = best.Strategy
    bc = curves[bname]
    bt = tradebooks[bname]

    base.yearly(bc).rename("return").to_csv(RESULTS / "yearly_best.csv")
    bc.to_csv(RESULTS / "equity_best.csv")
    bt.sort_values("ret", ascending=False).to_csv(RESULTS / "trades_best.csv", index=False)
    drawdown_episodes(bc).to_csv(RESULTS / "worst_drawdowns.csv", index=False)

    report = []
    report.append("# Minervini Korea — Price-only Phase Backtest\n")
    report.append(f"Data: {manifest['start']} to {manifest['end']}, {manifest['rows']:,} ticker-days, {manifest['tickers']:,} tickers.\n")
    report.append("## Best robust candidate by Calmar (>=20 trades)\n")
    for k in ["Strategy", "CAGR", "MDD", "Sharpe", "Calmar", "Final", "Trades", "WinRate", "AvgTrade", "MedianTrade", "AvgHoldDays", "ProfitFactor"]:
        v = best[k]
        report.append(f"- {k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"- {k}: {v}")
    report.append("\n## Top strategies\n")
    report.append(summary.head(15).to_markdown(index=False))
    report.append("\n\n## Annual returns — selected candidate\n")
    yr = base.yearly(bc)
    report.append("\n".join([f"- {idx.year}: {val:.2%}" for idx, val in yr.items()]))
    report.append("\n\n## Audit notes\n- D-close signal / D+1-open entry.\n- Historical sell tax schedule applied instead of a constant tax.\n- 20 bps one-way slippage + 1.5 bps commission.\n- Delisted history remains in marcap.\n- This phase does not yet include PIT sector membership or PIT fundamentals.\n")
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(summary.head(10).to_string(index=False), flush=True)
    print("BEST", bname, flush=True)


if __name__ == "__main__":
    main()
