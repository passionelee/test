from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as base
import run_backtest_fast as fast
import run_backtest_v3 as v3

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v4"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class P:
    name: str
    regime: str = "m200"
    rs: float = 85
    rsacc: float = 0
    breakvol: float = 1.2
    stop: float = 0.07
    max_pos: int = 5
    risk_exit: str = "none"  # none / nonwinner / all
    time_stop: int = 0        # sessions, 0 disables
    time_gain: float = 0.10   # require this peak gain by time_stop
    time_close_entry: bool = True
    ma20_fail_age: int = 0    # if >0, weak nonwinner below 20MA exits after age


def signal_mask(df: pd.DataFrame, p: P) -> np.ndarray:
    core = v3.P(
        name=p.name,
        rs=p.rs,
        rsacc=p.rsacc,
        vcp_shape="soft",
        breakvol=p.breakvol,
        stop=p.stop,
        full_stack=True,
        regime=p.regime,
    )
    return v3.signal_mask(df, core)


def build_signal_book(df: pd.DataFrame, p: P) -> dict:
    mask = signal_mask(df, p)
    cols = ["Date", "Code", "Name", "pivot20", "rs", "rsacc20", "breakvol"]
    sig = df.loc[mask, cols]
    if sig.empty:
        return {}
    return {pd.Timestamp(d): q.to_dict("records") for d, q in sig.groupby("Date", sort=False)}


def build_context(df: pd.DataFrame) -> dict:
    cols = [
        "Date", "Code", "Name", "Market", "adj_open", "adj_high", "adj_low", "adj_close",
        "ma20", "ma60", "midx", "mma20", "mma50", "mma200", "breadth_eligible"
    ]
    by_date = {}
    for d, q in df[cols].groupby("Date", sort=True):
        by_date[pd.Timestamp(d)] = q.drop(columns="Date").set_index("Code", drop=False)
    return {
        "dates": list(by_date.keys()),
        "by_date": by_date,
        "last_date": df.groupby("Code", sort=False).Date.max().to_dict(),
    }


def market_on(row, mode: str) -> bool:
    if mode == "m200":
        return bool(pd.notna(row.mma200) and row.midx > row.mma200)
    if mode == "fast":
        a = pd.notna(row.mma200) and row.midx > row.mma200
        b = pd.notna(row.mma50) and pd.notna(row.mma20) and row.midx > row.mma50 and row.mma20 > row.mma50
        return bool(a or b)
    return True


def simulate(df: pd.DataFrame, ctx: dict, p: P):
    dates = ctx["dates"]
    by_date = ctx["by_date"]
    last_date = ctx["last_date"]
    signal_book = build_signal_book(df, p)

    cash = 1.0
    pos = {}
    pending_entries = []
    pending_exits = {}  # code -> reason
    trades = []
    curve = []

    def sell(c, q, d, fill, reason):
        nonlocal cash
        tax = fast.sell_tax_rate(d, q["Market"])
        proceeds = q["shares"] * fill * (1 - base.SELL_COMM - tax)
        cash += proceeds
        trades.append({
            "Code": c, "Name": q["Name"], "Market": q["Market"],
            "entry": q["entry_date"], "exit": d,
            "entry_px": q["entry_px"], "exit_px": fill,
            "ret": proceeds / q["cost"] - 1,
            "reason": reason, "hold_days": (d - q["entry_date"]).days,
            "sessions": q["age"], "sell_tax": tax,
        })

    for d in dates:
        day = by_date[d]

        for c, q in pos.items():
            if c in day.index:
                q["last_close"] = float(day.at[c, "adj_close"])

        # Execute scheduled exits at next available open.
        done = []
        for c, reason in list(pending_exits.items()):
            if c not in pos or c not in day.index:
                continue
            row = day.loc[c]
            q = pos[c]
            fill = float(row.adj_open) * (1 - base.SLIP)
            sell(c, q, d, fill, reason)
            del pos[c]
            done.append(c)
        for c in done:
            pending_exits.pop(c, None)

        # Enter prior-close signals.
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
            score = float(sig["rs"]) + 0.35 * max(float(sig["rsacc20"]), 0.0) + 2 * min(float(sig["breakvol"]), 3.0)
            candidates.append((score, c, sig, row))
        candidates.sort(reverse=True, key=lambda z: z[0])

        slots = p.max_pos - len(pos)
        if slots > 0 and candidates and cash > 0:
            equity_before = cash + sum(q["shares"] * q["last_close"] for q in pos.values())
            target = equity_before / p.max_pos
            for _, c, sig, row in candidates[:slots]:
                alloc = min(target, cash)
                if alloc < equity_before * 0.03:
                    continue
                fill = float(row.adj_open) * (1 + base.SLIP)
                gross = fill * (1 + base.BUY_COMM)
                shares = alloc / gross
                cost = shares * gross
                cash -= cost
                pos[c] = {
                    "shares": shares, "entry_px": fill, "entry_date": d, "cost": cost,
                    "Name": sig["Name"], "Market": str(row.Market),
                    "peak": float(row.adj_high), "last_close": float(row.adj_close), "age": 0,
                }
        pending_entries = []

        stopped = []
        for c, q in list(pos.items()):
            if c not in day.index:
                continue
            row = day.loc[c]
            q["age"] += 1
            q["peak"] = max(q["peak"], float(row.adj_high))
            q["last_close"] = float(row.adj_close)
            peak_gain = q["peak"] / q["entry_px"] - 1

            # Initial hard stop until stock has proven itself.
            if peak_gain < 0.20:
                stop_px = q["entry_px"] * (1 - p.stop)
                if float(row.adj_low) <= stop_px:
                    raw = float(row.adj_open) if float(row.adj_open) < stop_px else stop_px
                    fill = raw * (1 - base.SLIP)
                    sell(c, q, d, fill, "stop")
                    stopped.append(c)
                    continue

            if d == last_date.get(c):
                sell(c, q, d, float(row.adj_close) * (1 - base.SLIP), "last_row")
                stopped.append(c)
                continue

            # Market regime deterioration: protect capital, but optionally retain proven winners.
            if p.risk_exit != "none" and not market_on(row, p.regime):
                if p.risk_exit == "all" or peak_gain < 0.20:
                    pending_exits.setdefault(c, "risk_off")
                    continue

            # Time stop: names that do not make progress should release capital.
            if p.time_stop and q["age"] >= p.time_stop and peak_gain < p.time_gain:
                weak = float(row.adj_close) < q["entry_px"] if p.time_close_entry else True
                if weak:
                    pending_exits.setdefault(c, "time_stop")
                    continue

            # Early failure below 20DMA before a +20% proof-of-strength move.
            if p.ma20_fail_age and q["age"] >= p.ma20_fail_age and peak_gain < 0.20:
                if pd.notna(row.ma20) and float(row.adj_close) < float(row.ma20):
                    pending_exits.setdefault(c, "ma20_fail")
                    continue

            # Winner management.
            if peak_gain >= 0.50:
                trail = row.ma60
            elif peak_gain >= 0.20:
                trail = row.ma20
            else:
                trail = np.nan
            if pd.notna(trail) and float(row.adj_close) < float(trail):
                pending_exits.setdefault(c, "trend")

        for c in stopped:
            pos.pop(c, None)
            pending_exits.pop(c, None)

        equity = cash + sum(q["shares"] * q["last_close"] for q in pos.values())
        curve.append((d, equity, len(pos), cash))
        pending_entries = signal_book.get(d, [])

    d = dates[-1]
    day = by_date[d]
    for c, q in list(pos.items()):
        px = float(day.at[c, "adj_close"]) if c in day.index else q["last_close"]
        sell(c, q, d, px * (1 - base.SLIP), "end")
    if curve:
        curve[-1] = (curve[-1][0], cash, 0, cash)
    return pd.DataFrame(curve, columns=["Date", "equity", "npos", "cash"]).set_index("Date"), pd.DataFrame(trades)


def tests():
    out = [
        P("m200_base"), P("fast_base", regime="fast"),
        P("m200_risk_nonwinner", risk_exit="nonwinner"),
        P("m200_risk_all", risk_exit="all"),
        P("fast_risk_nonwinner", regime="fast", risk_exit="nonwinner"),
        P("fast_risk_all", regime="fast", risk_exit="all"),
    ]
    for n in [10, 15, 20, 30, 40]:
        out.append(P(f"m200_time{n}", time_stop=n))
    for age in [5, 10, 15, 20]:
        out.append(P(f"m200_ma20fail{age}", ma20_fail_age=age))
    out += [
        P("m200_combo", risk_exit="nonwinner", time_stop=20, ma20_fail_age=10),
        P("fast_combo", regime="fast", risk_exit="nonwinner", time_stop=20, ma20_fail_age=10),
    ]
    for stop in [0.06, 0.07, 0.08, 0.09]:
        out.append(P(f"m200_combo_stop{int(stop*100)}", risk_exit="nonwinner", time_stop=20, ma20_fail_age=10, stop=stop))
    for mp in [3, 4, 5, 6]:
        out.append(P(f"m200_combo_pos{mp}", risk_exit="nonwinner", time_stop=20, ma20_fail_age=10, max_pos=mp))
    for gain in [0.05, 0.10, 0.15]:
        out.append(P(f"m200_timegain{int(gain*100)}", risk_exit="nonwinner", time_stop=20, time_gain=gain))
    return list({x.name: x for x in out}.values())


def main():
    print("Preparing V4 risk-management panel...", flush=True)
    df = v3.prepare(base.load_prepare())
    ctx = build_context(df)
    rows, curves, books = [], {}, {}
    params = tests()
    for i, p in enumerate(params, 1):
        signals = int(signal_mask(df, p).sum())
        print(f"[{i}/{len(params)}] {p.name} signals={signals}", flush=True)
        c, t = simulate(df, ctx, p)
        m = base.metrics(c, t)
        m.update({
            "Strategy": p.name, "Signals": signals, "regime": p.regime,
            "stop": p.stop, "max_pos": p.max_pos, "risk_exit": p.risk_exit,
            "time_stop": p.time_stop, "time_gain": p.time_gain,
            "ma20_fail_age": p.ma20_fail_age,
        })
        rows.append(m); curves[p.name] = c; books[p.name] = t
        pd.DataFrame(rows).to_csv(RESULTS / "checkpoint.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["Calmar", "CAGR"], ascending=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    robust = summary[summary.Trades >= 50]
    best = (robust if len(robust) else summary).iloc[0]
    name = best.Strategy
    base.yearly(curves[name]).rename("return").to_csv(RESULTS / "yearly_best.csv")
    curves[name].to_csv(RESULTS / "equity_best.csv")
    books[name].sort_values("ret", ascending=False).to_csv(RESULTS / "trades_best.csv", index=False)
    fast.drawdown_episodes(curves[name]).to_csv(RESULTS / "worst_drawdowns.csv", index=False)
    print(summary.head(20).to_string(index=False), flush=True)
    print("BEST", name, flush=True)


if __name__ == "__main__":
    main()
