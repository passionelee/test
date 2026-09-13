from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as base
import run_backtest_fast as fast
import run_backtest_v2 as v2

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v3"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class P:
    name: str
    rs: float = 85
    rsacc: float = 5
    vcp_shape: str = "soft"
    breakvol: float = 1.2
    stop: float = 0.07
    full_stack: bool = True
    ma_ratio: float = 0.95
    slope60_min: float = -0.01
    regime: str = "none"  # none / m200 / strict / fast / hybrid / breadth55
    near_high: bool = False


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = v2.prepare_v2(df)
    df = df.sort_values(["Code", "Date"]).copy()
    g = df.groupby("Code", sort=False)

    # 52-week price template.
    df["hi250"] = g.adj_high.transform(lambda s: s.rolling(250, min_periods=200).max())
    df["lo250"] = g.adj_low.transform(lambda s: s.rolling(250, min_periods=200).min())

    # Build point-in-time market proxies from prior-day market-cap weights.
    df["ret1"] = np.expm1(df.logret)
    df["prev_mcap"] = g.Marcap.shift(1)
    tmp = df.dropna(subset=["prev_mcap", "ret1"]).copy()
    tmp["wr"] = tmp.ret1 * tmp.prev_mcap
    daily = tmp.groupby(["Date", "Market"], sort=True).agg(wr=("wr", "sum"), w=("prev_mcap", "sum")).reset_index()
    daily["mret"] = daily.wr / daily.w
    daily = daily.sort_values(["Market", "Date"])
    daily["midx"] = daily.groupby("Market", sort=False).mret.transform(lambda s: (1 + s.fillna(0)).cumprod())
    gm = daily.groupby("Market", sort=False).midx
    for w in [20, 50, 100, 200]:
        daily[f"mma{w}"] = gm.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())

    df = df.merge(daily[["Date", "Market", "midx", "mma20", "mma50", "mma100", "mma200"]], on=["Date", "Market"], how="left")
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    return df


def vcp_mask(df: pd.DataFrame, shape: str) -> np.ndarray:
    return v2.vcp_mask(df, shape)


def regime_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(df), dtype=bool)
    if mode == "m200":
        return (df.midx > df.mma200).fillna(False).to_numpy()
    if mode == "strict":
        return ((df.midx > df.mma200) & (df.mma50 > df.mma200)).fillna(False).to_numpy()
    if mode == "fast":
        return (
            (df.midx > df.mma200)
            | ((df.midx > df.mma50) & (df.mma20 > df.mma50))
        ).fillna(False).to_numpy()
    if mode == "breadth55":
        return (df.breadth_eligible >= 0.55).fillna(False).to_numpy()
    if mode == "hybrid":
        return (
            ((df.midx > df.mma200) & (df.mma50 > df.mma200))
            | ((df.midx > df.mma50) & (df.mma20 > df.mma50) & (df.breadth_eligible >= 0.55))
        ).fillna(False).to_numpy()
    raise ValueError(mode)


def signal_mask(df: pd.DataFrame, p: P) -> np.ndarray:
    liquid = ((df.adv20 >= base.MIN_ADV20) & (df.Marcap >= base.MIN_MCAP)).to_numpy()
    if p.full_stack:
        trend = (
            (df.adj_close > df.ma20)
            & (df.ma20 > df.ma60)
            & (df.ma60 > df.ma120)
            & (df.ma120 > df.ma200)
            & (df.slope20 > 0)
            & (df.slope60 > 0)
        ).to_numpy()
    else:
        trend = (
            (df.adj_close > df.ma20)
            & (df.ma20 > df.ma60)
            & (df.slope20 > 0)
            & (df.slope60 >= p.slope60_min)
            & ((df.ma60 / df.ma120) >= p.ma_ratio)
        ).to_numpy()

    leader = ((df.rs >= p.rs) & (df.rsacc20 >= p.rsacc)).to_numpy()
    vcp = vcp_mask(df, p.vcp_shape)
    breakout = ((df.adj_close > df.pivot20) & (df.breakvol >= p.breakvol)).to_numpy()
    regime = regime_mask(df, p.regime)
    if p.near_high:
        template = (
            (df.adj_close >= 0.75 * df.hi250)
            & (df.adj_close >= 1.30 * df.lo250)
        ).fillna(False).to_numpy()
    else:
        template = np.ones(len(df), dtype=bool)
    return liquid & trend & leader & vcp & breakout & regime & template


def build_signal_book(df: pd.DataFrame, p: P) -> dict:
    mask = signal_mask(df, p)
    cols = ["Date", "Code", "Name", "pivot20", "rs", "rsacc20", "breakvol"]
    sig = df.loc[mask, cols]
    if sig.empty:
        return {}
    return {pd.Timestamp(d): q.to_dict("records") for d, q in sig.groupby("Date", sort=False)}


def tests() -> list[P]:
    out = [
        P("confirmed_none"),
        P("confirmed_m200", regime="m200"),
        P("confirmed_strict", regime="strict"),
        P("confirmed_fast", regime="fast"),
        P("confirmed_hybrid", regime="hybrid"),
        P("confirmed_breadth55", regime="breadth55"),
        P("confirmed_m200_high", regime="m200", near_high=True),
        P("confirmed_fast_high", regime="fast", near_high=True),
        P("early_m200", full_stack=False, regime="m200"),
        P("early_fast", full_stack=False, regime="fast"),
    ]
    # Sensitivity around the two most plausible market regimes.
    for mode in ["m200", "fast"]:
        for rs in [80, 85, 90, 95]:
            out.append(P(f"{mode}_rs{rs}", regime=mode, rs=rs))
        for acc in [0, 5, 10]:
            out.append(P(f"{mode}_acc{acc}", regime=mode, rsacc=acc))
        for bv in [1.0, 1.2, 1.4]:
            out.append(P(f"{mode}_bv{bv:.1f}", regime=mode, breakvol=bv))
        for st in [0.05, 0.07, 0.09]:
            out.append(P(f"{mode}_stop{int(st*100)}", regime=mode, stop=st))
    return list({p.name: p for p in out}.values())


def main():
    print("Preparing V3 market-regime panel...", flush=True)
    df = prepare(base.load_prepare())
    ctx = fast.build_context(df)
    fast.build_signal_book = build_signal_book

    rows = []
    curves = {}
    books = {}
    params = tests()
    for i, p in enumerate(params, 1):
        signals = int(signal_mask(df, p).sum())
        print(f"[{i}/{len(params)}] {p.name} signals={signals}", flush=True)
        c, t = fast.simulate_fast(df, ctx, p)
        m = base.metrics(c, t)
        m.update({
            "Strategy": p.name,
            "Signals": signals,
            "rs": p.rs,
            "rsacc": p.rsacc,
            "vcp_shape": p.vcp_shape,
            "breakvol": p.breakvol,
            "stop": p.stop,
            "full_stack": p.full_stack,
            "regime": p.regime,
            "near_high": p.near_high,
        })
        rows.append(m)
        curves[p.name] = c
        books[p.name] = t
        pd.DataFrame(rows).to_csv(RESULTS / "checkpoint.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["Calmar", "CAGR"], ascending=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    robust = summary[summary.Trades >= 50]
    if robust.empty:
        robust = summary[summary.Trades >= 20]
    best = (robust if len(robust) else summary).iloc[0]
    name = best.Strategy
    curve = curves[name]
    book = books[name]
    base.yearly(curve).rename("return").to_csv(RESULTS / "yearly_best.csv")
    curve.to_csv(RESULTS / "equity_best.csv")
    book.sort_values("ret", ascending=False).to_csv(RESULTS / "trades_best.csv", index=False)
    fast.drawdown_episodes(curve).to_csv(RESULTS / "worst_drawdowns.csv", index=False)

    print(summary.head(20).to_string(index=False), flush=True)
    print("BEST", name, flush=True)


if __name__ == "__main__":
    main()
