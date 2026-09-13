from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_backtest as base
import run_backtest_fast as fast
import run_backtest_v3 as v3

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)


def p(name, regime="m200", rs=85, rsacc=0, breakvol=1.2, stop=0.07):
    return v3.P(name=name, regime=regime, rs=rs, rsacc=rsacc,
                breakvol=breakvol, stop=stop, full_stack=True,
                vcp_shape="soft")


def signal_book(df, param):
    mask = v3.signal_mask(df, param)
    cols = ["Date", "Code", "Name", "pivot20", "rs", "rsacc20", "breakvol"]
    sig = df.loc[mask, cols]
    return {} if sig.empty else {pd.Timestamp(d): q.to_dict("records") for d, q in sig.groupby("Date", sort=False)}


def run_one(df, param, slip=0.002):
    old = base.SLIP
    base.SLIP = slip
    fast.build_signal_book = signal_book
    try:
        ctx = fast.build_context(df)
        c, t = fast.simulate_fast(df, ctx, param)
        m = base.metrics(c, t)
    finally:
        base.SLIP = old
    return c, t, m


def temporal_splits(df, param):
    periods = [
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("2019-2022", "2019-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-09-11"),
        ("2015-2020", "2015-01-01", "2020-12-31"),
        ("2021-2026", "2021-01-01", "2026-09-11"),
    ]
    rows=[]
    for name, a, b in periods:
        sub=df[(df.Date>=a)&(df.Date<=b)].copy()
        c,t,m=run_one(sub,param)
        rows.append({"period":name, **m})
    return pd.DataFrame(rows)


def market_benchmarks(df):
    out=[]
    for market in ["KOSPI","KOSDAQ"]:
        s=(df[df.Market==market][["Date","midx"]]
           .drop_duplicates("Date").dropna().sort_values("Date").set_index("Date").midx)
        if len(s)<2: continue
        eq=s/s.iloc[0]
        yrs=(eq.index[-1]-eq.index[0]).days/365.25
        cagr=eq.iloc[-1]**(1/yrs)-1
        mdd=(eq/eq.cummax()-1).min()
        dr=eq.pct_change().fillna(0)
        sharpe=np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else np.nan
        out.append({"benchmark":f"synthetic_{market}","CAGR":cagr,"MDD":mdd,"Sharpe":sharpe,"Final":eq.iloc[-1]})
    return pd.DataFrame(out)


def block_bootstrap(curve, n=1000, block=20, seed=42):
    r=curve.equity.pct_change().dropna().to_numpy()
    rng=np.random.default_rng(seed)
    rows=[]
    N=len(r)
    starts=np.arange(max(N-block+1,1))
    for i in range(n):
        samp=[]
        while len(samp)<N:
            st=int(rng.choice(starts))
            samp.extend(r[st:st+block])
        rr=np.array(samp[:N])
        eq=np.cumprod(1+rr)
        yrs=N/252
        cagr=eq[-1]**(1/yrs)-1
        dd=eq/np.maximum.accumulate(eq)-1
        rows.append((cagr,dd.min()))
    z=pd.DataFrame(rows,columns=["CAGR","MDD"])
    q=z.quantile([0.05,0.25,0.50,0.75,0.95])
    q.index.name="quantile"
    return q,z


def concentration(trades):
    if trades.empty: return {}
    pos=trades.loc[trades.ret>0,"ret"].sort_values(ascending=False)
    gross=pos.sum()
    return {
        "trades":int(len(trades)),
        "positive_trades":int(len(pos)),
        "top1_share_of_gross_positive": float(pos.head(1).sum()/gross) if gross else np.nan,
        "top3_share_of_gross_positive": float(pos.head(3).sum()/gross) if gross else np.nan,
        "top5_share_of_gross_positive": float(pos.head(5).sum()/gross) if gross else np.nan,
        "top10_share_of_gross_positive": float(pos.head(10).sum()/gross) if gross else np.nan,
        "largest_trade": float(pos.iloc[0]) if len(pos) else np.nan,
    }


def main():
    print("Preparing V5 validation panel...",flush=True)
    df=v3.prepare(base.load_prepare())
    mainp=p("m200_acc0")
    fastp=p("fast_acc0",regime="fast")

    # Base candidates.
    c,t,m=run_one(df,mainp)
    cf,tf,mf=run_one(df,fastp)
    pd.DataFrame([{"Strategy":"m200_acc0",**m},{"Strategy":"fast_acc0",**mf}]).to_csv(RESULTS/"base_candidates.csv",index=False)
    temporal_splits(df,mainp).to_csv(RESULTS/"temporal_m200.csv",index=False)
    temporal_splits(df,fastp).to_csv(RESULTS/"temporal_fast.csv",index=False)
    market_benchmarks(df).to_csv(RESULTS/"benchmarks.csv",index=False)

    # Cost stress.
    costs=[]
    for strat,param in [("m200",mainp),("fast",fastp)]:
        for slip in [0.002,0.004,0.006,0.010]:
            cc,tt,mm=run_one(df,param,slip=slip)
            costs.append({"Strategy":strat,"one_way_slippage":slip,**mm})
    pd.DataFrame(costs).to_csv(RESULTS/"cost_stress.csv",index=False)

    # Parameter neighbourhood around the frozen structure.
    grid=[]
    for regime in ["m200","fast"]:
      for rs in [80,85,90]:
       for bv in [1.0,1.2,1.4]:
        for stop in [0.06,0.07,0.08]:
            pp=p(f"{regime}_rs{rs}_bv{bv}_s{stop}",regime=regime,rs=rs,breakvol=bv,stop=stop)
            cc,tt,mm=run_one(df,pp)
            grid.append({"regime":regime,"rs":rs,"breakvol":bv,"stop":stop,**mm})
            print("grid",regime,rs,bv,stop,mm.get("CAGR"),mm.get("MDD"),flush=True)
    g=pd.DataFrame(grid).sort_values(["Calmar","CAGR"],ascending=False)
    g.to_csv(RESULTS/"parameter_grid.csv",index=False)

    # Winner concentration and bootstrap of the frozen candidate.
    (RESULTS/"winner_concentration.json").write_text(json.dumps(concentration(t),indent=2,ensure_ascii=False),encoding="utf-8")
    q,z=block_bootstrap(c,n=1000,block=20,seed=42)
    q.to_csv(RESULTS/"bootstrap_quantiles.csv")
    z.to_csv(RESULTS/"bootstrap_samples.csv",index=False)

    # Summarize grid stability.
    stable={
        "grid_count":int(len(g)),
        "positive_cagr_fraction":float((g.CAGR>0).mean()),
        "mdd_better_than_40_fraction":float((g.MDD>-0.40).mean()),
        "cagr_median":float(g.CAGR.median()),
        "cagr_q25":float(g.CAGR.quantile(.25)),
        "cagr_q75":float(g.CAGR.quantile(.75)),
        "mdd_median":float(g.MDD.median()),
        "calmar_median":float(g.Calmar.median()),
        "best_by_calmar":g.iloc[0].to_dict(),
    }
    (RESULTS/"grid_stability.json").write_text(json.dumps(stable,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print("BASE",m,flush=True)
    print("FAST",mf,flush=True)
    print("STABILITY",stable,flush=True)
    print("CONCENTRATION",concentration(t),flush=True)
    print("BOOTSTRAP\n",q.to_string(),flush=True)

if __name__=="__main__":
    main()
