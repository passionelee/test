from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as base
import run_backtest_fast as fast

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v2"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class P:
    name: str
    rs: float = 85
    rsacc: float = 5
    ma_ratio: float = 0.95
    slope60_min: float = -0.01
    vcp_shape: str = "soft"  # soft / medium / strict / none
    breakvol: float = 1.2
    stop: float = 0.07
    breadth: float | None = None
    full_stack: bool = False


def prepare_v2(df: pd.DataFrame) -> pd.DataFrame:
    # Correct breadth: denominator is only liquid/market-cap eligible stocks.
    eligible = (df.adv20 >= base.MIN_ADV20) & (df.Marcap >= base.MIN_MCAP)
    above = (df.adj_close > df.ma60) & eligible
    num = above.groupby(df.Date).sum()
    den = eligible.groupby(df.Date).sum().replace(0, np.nan)
    breadth = (num / den).rename("breadth_eligible")
    df = df.drop(columns=["breadth_eligible"], errors="ignore").join(breadth, on="Date")

    # VCP is a pre-breakout setup. Use D-1 features, never the breakout day's expansion.
    df = df.sort_values(["Code", "Date"]).copy()
    g = df.groupby("Code", sort=False)
    for c in ["range10", "range30", "range60", "atr10", "atr30", "vol10", "vol50"]:
        df[c + "_l1"] = g[c].shift(1)
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    return df


def vcp_mask(df: pd.DataFrame, shape: str) -> np.ndarray:
    if shape == "none":
        return np.ones(len(df), dtype=bool)
    if shape == "strict":
        c30, c10, last, atr, vol = 0.82, 0.78, 0.12, 0.85, 0.80
    elif shape == "medium":
        c30, c10, last, atr, vol = 0.90, 0.85, 0.14, 0.90, 0.90
    else:
        c30, c10, last, atr, vol = 0.95, 0.90, 0.16, 0.95, 1.00
    return (
        (df.range30_l1.to_numpy() <= c30 * df.range60_l1.to_numpy())
        & (df.range10_l1.to_numpy() <= c10 * df.range30_l1.to_numpy())
        & (df.range10_l1.to_numpy() <= last)
        & ((df.atr10_l1 / df.atr30_l1).to_numpy() <= atr)
        & ((df.vol10_l1 / df.vol50_l1).to_numpy() <= vol)
    )


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
    regime = np.ones(len(df), dtype=bool) if p.breadth is None else (df.breadth_eligible >= p.breadth).to_numpy()
    return liquid & trend & leader & vcp & breakout & regime


def build_signal_book(df: pd.DataFrame, p: P) -> dict:
    mask = signal_mask(df, p)
    cols = ["Date", "Code", "Name", "pivot20", "rs", "rsacc20", "breakvol"]
    sig = df.loc[mask, cols]
    if sig.empty:
        return {}
    return {pd.Timestamp(d): q.to_dict("records") for d, q in sig.groupby("Date", sort=False)}


def tests() -> list[P]:
    out = [
        P("confirmed_soft", full_stack=True, vcp_shape="soft"),
        P("early_soft_no_accel", rsacc=-100, vcp_shape="soft"),
        P("early_soft_accel0", rsacc=0, vcp_shape="soft"),
        P("early_soft_accel5", rsacc=5, vcp_shape="soft"),
        P("early_soft_accel10", rsacc=10, vcp_shape="soft"),
        P("early_medium", vcp_shape="medium"),
        P("early_strict", vcp_shape="strict"),
        P("early_no_vcp", vcp_shape="none"),
        P("early_soft_breadth40", breadth=0.40),
        P("early_soft_breadth45", breadth=0.45),
        P("early_soft_breadth50", breadth=0.50),
    ]
    for rs in [80, 85, 90]:
        out.append(P(f"rs{rs}", rs=rs))
    for r in [0.92, 0.95, 0.98]:
        out.append(P(f"maratio{r:.2f}", ma_ratio=r))
    for s in [-0.02, -0.01, 0.0]:
        out.append(P(f"slope60_{s:+.2f}", slope60_min=s))
    for bv in [1.0, 1.2, 1.4]:
        out.append(P(f"breakvol{bv:.1f}", breakvol=bv))
    for st in [0.05, 0.07, 0.09]:
        out.append(P(f"stop{int(st*100)}", stop=st))
    return list({p.name: p for p in out}.values())


def main():
    print("Loading base panel...", flush=True)
    df = prepare_v2(base.load_prepare())
    ctx = fast.build_context(df)
    params = tests()

    # Inject the corrected V2 signal builder into the already audited fast portfolio engine.
    fast.build_signal_book = build_signal_book

    rows = []
    curves = {}
    books = {}
    for i, p in enumerate(params, 1):
        sig_count = int(signal_mask(df, p).sum())
        print(f"[{i}/{len(params)}] {p.name} signals={sig_count}", flush=True)
        c, t = fast.simulate_fast(df, ctx, p)
        m = base.metrics(c, t)
        m.update({
            "Strategy": p.name,
            "Signals": sig_count,
            "rs": p.rs,
            "rsacc": p.rsacc,
            "ma_ratio": p.ma_ratio,
            "slope60_min": p.slope60_min,
            "vcp_shape": p.vcp_shape,
            "breakvol": p.breakvol,
            "stop": p.stop,
            "breadth": p.breadth,
            "full_stack": p.full_stack,
        })
        rows.append(m)
        curves[p.name] = c
        books[p.name] = t
        pd.DataFrame(rows).to_csv(RESULTS / "checkpoint.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["Calmar", "CAGR"], ascending=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    # Require a minimum evidence base; otherwise a tiny sample can win Calmar by accident.
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

    # Structural comparison table for the core hypotheses.
    core_names = [
        "confirmed_soft", "early_soft_no_accel", "early_soft_accel0",
        "early_soft_accel5", "early_soft_accel10", "early_medium",
        "early_strict", "early_no_vcp", "early_soft_breadth40",
        "early_soft_breadth45", "early_soft_breadth50"
    ]
    summary[summary.Strategy.isin(core_names)].to_csv(RESULTS / "structural_ablation.csv", index=False)

    manifest = {
        "rows": int(len(df)),
        "tickers": int(df.Code.nunique()),
        "start": str(df.Date.min().date()),
        "end": str(df.Date.max().date()),
        "changes_from_v1": [
            "VCP/ATR/volume contraction evaluated on D-1, not breakout day D",
            "breadth denominator restricted to liquid/market-cap eligible names",
            "Early Trend allows flat-to-mildly-declining 60DMA before full Stage-2 confirmation",
            "RS acceleration, VCP strictness, MA ratio, 60DMA slope, breakout volume, stop and breadth sensitivity retested",
        ],
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.head(15).to_string(index=False), flush=True)
    print("BEST", name, flush=True)


if __name__ == "__main__":
    main()
