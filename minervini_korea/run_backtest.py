from __future__ import annotations

import json, math, os, shutil, subprocess
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "external"
DATA_REPO = EXT / "marcap"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-09-11")
WARMUP = pd.Timestamp("2014-01-01")
SLIP = 0.002
BUY_COMM = 0.00015
SELL_COMM = 0.00015
SELL_TAX = 0.0018
MAX_POS = 5
MIN_ADV20 = 5_000_000_000
MIN_MCAP = 100_000_000_000


def sh(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_data():
    if DATA_REPO.exists():
        return
    EXT.mkdir(exist_ok=True)
    sh(["git", "clone", "--depth", "1", "https://github.com/FinanceData/marcap.git", str(DATA_REPO)])


def classify_common(df: pd.DataFrame) -> pd.Series:
    code = df["Code"].astype(str).str.zfill(6)
    name = df["Name"].fillna("").astype(str)
    market = df["Market"].fillna("").astype(str)
    etf_prefix = r"^(KODEX|TIGER|RISE|ACE|ARIRANG|HANARO|SOL|KOSEF|KBSTAR|PLUS|TIMEFOLIO|WOORI|히어로즈)"
    bad_name = name.str.contains(r"스팩|SPAC|리츠|REIT|ETN", case=False, regex=True) | name.str.contains(etf_prefix, case=False, regex=True)
    return market.isin(["KOSPI", "KOSDAQ"]) & code.str.len().eq(6) & code.str[-1].eq("0") & (~bad_name)


def rolling_by_code(df, col, window, func="mean"):
    g = df.groupby("Code", sort=False)[col]
    if func == "mean":
        return g.transform(lambda s: s.rolling(window, min_periods=window).mean())
    if func == "max":
        return g.transform(lambda s: s.rolling(window, min_periods=window).max())
    if func == "min":
        return g.transform(lambda s: s.rolling(window, min_periods=window).min())
    raise ValueError(func)


def load_prepare():
    ensure_data()
    frames = []
    for y in range(2014, 2027):
        p = DATA_REPO / "data" / f"marcap-{y}.parquet"
        if not p.exists():
            print("missing", p)
            continue
        d = pd.read_parquet(p)
        keep = [c for c in ["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChagesRatio","Marcap","Market"] if c in d.columns]
        d = d[keep].copy()
        frames.append(d)
    if not frames:
        raise RuntimeError("No marcap parquet files found")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= WARMUP) & (df["Date"] <= END)].copy()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df = df[classify_common(df)].copy()
    for c in ["Open","High","Low","Close","Volume","Amount","Marcap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.Close > 0) & (df.Open >= 0) & (df.High > 0) & (df.Low > 0)].copy()
    df.sort_values(["Code","Date"], inplace=True)

    # Exchange-provided daily percentage change is used to build a corporate-action-continuous
    # synthetic price series. If missing, fall back to raw close pct_change.
    if "ChagesRatio" in df.columns:
        r = pd.to_numeric(df["ChagesRatio"], errors="coerce") / 100.0
    else:
        r = df.groupby("Code", sort=False).Close.pct_change(fill_method=None)
    raw_r = df.groupby("Code", sort=False).Close.pct_change(fill_method=None)
    r = r.where(r.notna(), raw_r).clip(lower=-0.95, upper=3.0).fillna(0.0)
    df["logret"] = np.log1p(r)
    df["cumlog"] = df.groupby("Code", sort=False).logret.cumsum()
    df["adj_close"] = 100.0 * np.exp(df.cumlog)
    scale = (df.adj_close / df.Close).replace([np.inf,-np.inf], np.nan)
    df["adj_open"] = df.Open * scale
    df["adj_high"] = df.High * scale
    df["adj_low"] = df.Low * scale

    # Indicators
    for w in [10,20,30,50,60,120,200]:
        if w in [20,60,120,200]:
            df[f"ma{w}"] = rolling_by_code(df, "adj_close", w, "mean")
    g = df.groupby("Code", sort=False)
    for w in [20,60,120,250]:
        df[f"r{w}"] = g.adj_close.pct_change(w, fill_method=None)
    df["ma20_prev10"] = g.ma20.shift(10)
    df["ma60_prev20"] = g.ma60.shift(20)
    df["slope20"] = df.ma20 / df.ma20_prev10 - 1
    df["slope60"] = df.ma60 / df.ma60_prev20 - 1

    prev_close = g.adj_close.shift(1)
    tr = pd.concat([(df.adj_high-df.adj_low).abs(), (df.adj_high-prev_close).abs(), (df.adj_low-prev_close).abs()], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr10"] = df.groupby("Code", sort=False).tr.transform(lambda s: s.rolling(10, min_periods=10).mean())
    df["atr30"] = df.groupby("Code", sort=False).tr.transform(lambda s: s.rolling(30, min_periods=30).mean())
    for w in [10,30,60]:
        hi = df.groupby("Code", sort=False).adj_high.transform(lambda s, w=w: s.rolling(w, min_periods=w).max())
        lo = df.groupby("Code", sort=False).adj_low.transform(lambda s, w=w: s.rolling(w, min_periods=w).min())
        df[f"range{w}"] = (hi-lo)/hi
    for w in [10,20,50]:
        df[f"vol{w}"] = df.groupby("Code", sort=False).Volume.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["adv20"] = df.groupby("Code", sort=False).Amount.transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["pivot20"] = df.groupby("Code", sort=False).adj_high.transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    df["breakvol"] = df.Volume / df.vol20

    # Cross-sectional relative strength; percentile 0..100.
    df["rsraw"] = 0.15*df.r20 + 0.30*df.r60 + 0.30*df.r120 + 0.25*df.r250
    df["rs"] = 100.0 * df.groupby("Date", sort=False).rsraw.rank(pct=True)
    df["rs20ago"] = df.groupby("Code", sort=False).rs.shift(20)
    df["rsacc20"] = df.rs - df.rs20ago

    eligible_basic = (df.adv20 >= MIN_ADV20) & (df.Marcap >= MIN_MCAP)
    above60 = (df.adj_close > df.ma60) & eligible_basic
    breadth = above60.groupby(df.Date).mean().rename("breadth60")
    df = df.join(breadth, on="Date")

    df = df[(df.Date >= START) & (df.Date <= END)].copy()
    df.sort_values(["Date","Code"], inplace=True)
    return df


@dataclass(frozen=True)
class Params:
    name: str
    rs: float = 90
    rsacc: float = 10
    ma_ratio: float = 0.97
    last_range: float = 0.12
    vol_dry: float = 0.80
    atr_ratio: float = 0.85
    breakvol: float = 1.30
    stop: float = 0.06
    breadth: float | None = None
    full_stack: bool = False
    use_vcp: bool = True


def signal_mask(df, p: Params):
    liquid = (df.adv20 >= MIN_ADV20) & (df.Marcap >= MIN_MCAP)
    if p.full_stack:
        trend = (df.adj_close > df.ma20) & (df.ma20 > df.ma60) & (df.ma60 > df.ma120) & (df.ma120 > df.ma200) & (df.slope20 > 0) & (df.slope60 > 0)
    else:
        trend = (df.adj_close > df.ma20) & (df.ma20 > df.ma60) & (df.slope20 > 0) & (df.slope60 > 0) & ((df.ma60/df.ma120) >= p.ma_ratio)
    leader = (df.rs >= p.rs) & (df.rsacc20 >= p.rsacc)
    vcp = (df.range30 <= 0.82*df.range60) & (df.range10 <= 0.78*df.range30) & (df.range10 <= p.last_range) & ((df.atr10/df.atr30) <= p.atr_ratio) & ((df.vol10/df.vol50) <= p.vol_dry)
    if not p.use_vcp:
        vcp = pd.Series(True, index=df.index)
    breakout = (df.adj_close > df.pivot20) & (df.breakvol >= p.breakvol)
    regime = pd.Series(True, index=df.index) if p.breadth is None else (df.breadth60 >= p.breadth)
    return liquid & trend & leader & vcp & breakout & regime


def simulate(df: pd.DataFrame, p: Params):
    x = df[["Date","Code","Name","adj_open","adj_high","adj_low","adj_close","ma20","ma60","pivot20","rs","rsacc20","breakvol"]].copy()
    x["signal"] = signal_mask(df, p).values
    by_date = {d: q for d,q in x.groupby("Date", sort=True)}
    dates = sorted(by_date)
    last_date = df.groupby("Code").Date.max().to_dict()
    cash = 1.0
    pos = {}
    pending_entries = []
    pending_exits = set()
    trades=[]
    curve=[]
    last_px={}

    for d in dates:
        day = by_date[d].set_index("Code", drop=False)
        for c,row in day.iterrows():
            last_px[c]=float(row.adj_close)

        # exits scheduled by prior close
        for c in list(pending_exits):
            if c not in pos or c not in day.index:
                continue
            row=day.loc[c]
            fill=float(row.adj_open)*(1-SLIP)
            q=pos[c]
            proceeds=q["shares"]*fill*(1-SELL_COMM-SELL_TAX)
            cash += proceeds
            pnl=proceeds/q["cost"]-1
            trades.append({"Code":c,"Name":q["Name"],"entry":q["entry_date"],"exit":d,"entry_px":q["entry_px"],"exit_px":fill,"ret":pnl,"reason":"trend","hold_days":(d-q["entry_date"]).days})
            del pos[c]
        pending_exits.clear()

        # new entries from prior-close signals
        candidates=[]
        for sig in pending_entries:
            c=sig["Code"]
            if c in pos or c not in day.index:
                continue
            row=day.loc[c]
            gap=float(row.adj_open)/sig["pivot20"]-1 if sig["pivot20"]>0 else 9
            if gap > 0.05:
                continue
            score=sig["rs"] + 0.35*max(sig["rsacc20"],0) + 2.0*min(sig["breakvol"],3)
            candidates.append((score,c,sig,row))
        candidates.sort(reverse=True, key=lambda z:z[0])
        slots=MAX_POS-len(pos)
        if slots>0 and candidates and cash>0:
            # reserve equal target slots; unused cash is allowed
            equity_before=cash+sum(q["shares"]*last_px.get(c,q["entry_px"]) for c,q in pos.items())
            target=equity_before/MAX_POS
            for _,c,sig,row in candidates[:slots]:
                alloc=min(target,cash)
                if alloc < equity_before*0.03:
                    continue
                fill=float(row.adj_open)*(1+SLIP)
                gross_per_share=fill*(1+BUY_COMM)
                shares=alloc/gross_per_share
                cost=shares*gross_per_share
                cash-=cost
                pos[c]={"shares":shares,"entry_px":fill,"entry_date":d,"cost":cost,"Name":sig["Name"],"peak":float(row.adj_high)}
        pending_entries=[]

        # intraday hard stop and end-of-day trailing decision
        stopped=[]
        for c,q in list(pos.items()):
            if c not in day.index:
                continue
            row=day.loc[c]
            q["peak"]=max(q["peak"],float(row.adj_high))
            peak_gain=q["peak"]/q["entry_px"]-1
            if peak_gain < 0.20:
                stop_px=q["entry_px"]*(1-p.stop)
                if float(row.adj_low) <= stop_px:
                    raw_fill=float(row.adj_open) if float(row.adj_open) < stop_px else stop_px
                    fill=raw_fill*(1-SLIP)
                    proceeds=q["shares"]*fill*(1-SELL_COMM-SELL_TAX)
                    cash+=proceeds
                    pnl=proceeds/q["cost"]-1
                    trades.append({"Code":c,"Name":q["Name"],"entry":q["entry_date"],"exit":d,"entry_px":q["entry_px"],"exit_px":fill,"ret":pnl,"reason":"stop","hold_days":(d-q["entry_date"]).days})
                    stopped.append(c)
                    continue
            # last available row: liquidate rather than carry a ghost position
            if d == last_date.get(c):
                fill=float(row.adj_close)*(1-SLIP)
                proceeds=q["shares"]*fill*(1-SELL_COMM-SELL_TAX)
                cash+=proceeds
                pnl=proceeds/q["cost"]-1
                trades.append({"Code":c,"Name":q["Name"],"entry":q["entry_date"],"exit":d,"entry_px":q["entry_px"],"exit_px":fill,"ret":pnl,"reason":"last_row","hold_days":(d-q["entry_date"]).days})
                stopped.append(c)
                continue
            # Winners get more room as they become exceptional.
            if peak_gain >= 0.50:
                trail=row.ma60
            elif peak_gain >= 0.20:
                trail=row.ma20
            else:
                trail=np.nan
            if pd.notna(trail) and float(row.adj_close) < float(trail):
                pending_exits.add(c)
        for c in stopped:
            pos.pop(c,None)
            pending_exits.discard(c)

        equity=cash+sum(q["shares"]*last_px.get(c,q["entry_px"]) for c,q in pos.items())
        curve.append((d,equity,len(pos),cash))

        sigs=day[day.signal]
        if len(sigs):
            pending_entries=[{"Code":r.Code,"Name":r.Name,"pivot20":float(r.pivot20),"rs":float(r.rs),"rsacc20":float(r.rsacc20),"breakvol":float(r.breakvol)} for _,r in sigs.iterrows()]

    # liquidate any remaining at final close
    d=dates[-1]
    day=by_date[d].set_index("Code", drop=False)
    for c,q in list(pos.items()):
        px=last_px.get(c,q["entry_px"])
        fill=px*(1-SLIP)
        proceeds=q["shares"]*fill*(1-SELL_COMM-SELL_TAX)
        cash+=proceeds
        pnl=proceeds/q["cost"]-1
        trades.append({"Code":c,"Name":q["Name"],"entry":q["entry_date"],"exit":d,"entry_px":q["entry_px"],"exit_px":fill,"ret":pnl,"reason":"end","hold_days":(d-q["entry_date"]).days})
        del pos[c]
    if curve:
        curve[-1]=(curve[-1][0],cash,0,cash)
    curve=pd.DataFrame(curve,columns=["Date","equity","npos","cash"]).set_index("Date")
    tr=pd.DataFrame(trades)
    return curve,tr


def metrics(curve,trades):
    if len(curve)<2:
        return {}
    years=(curve.index[-1]-curve.index[0]).days/365.25
    cagr=curve.equity.iloc[-1]**(1/years)-1
    peak=curve.equity.cummax()
    dd=curve.equity/peak-1
    mdd=dd.min()
    dret=curve.equity.pct_change().fillna(0)
    sharpe=np.sqrt(252)*dret.mean()/dret.std() if dret.std()>0 else np.nan
    calmar=cagr/abs(mdd) if mdd<0 else np.nan
    if len(trades):
        win=(trades.ret>0).mean()
        avg=trades.ret.mean()
        med=trades.ret.median()
        avg_hold=trades.hold_days.mean()
        pf=trades.loc[trades.ret>0,"ret"].sum()/abs(trades.loc[trades.ret<0,"ret"].sum()) if (trades.ret<0).any() else np.inf
    else:
        win=avg=med=avg_hold=pf=np.nan
    return dict(CAGR=cagr,MDD=mdd,Sharpe=sharpe,Calmar=calmar,Final=curve.equity.iloc[-1],Trades=len(trades),WinRate=win,AvgTrade=avg,MedianTrade=med,AvgHoldDays=avg_hold,ProfitFactor=pf)


def yearly(curve):
    e=curve.equity
    out=e.resample("YE").last().pct_change()
    if len(out):
        # first year return from initial 1.0
        out.iloc[0]=e[e.index.year==e.index[0].year].iloc[-1]/1.0-1
    return out


def main():
    df=load_prepare()
    manifest={"rows":int(len(df)),"tickers":int(df.Code.nunique()),"start":str(df.Date.min().date()),"end":str(df.Date.max().date()),"source":"FinanceData/marcap","notes":["Common-stock filter approximated from market/code/name.","Corporate-action-continuous synthetic OHLC built from exchange daily change ratio.","Sector and point-in-time fundamental filters are not included in this price-only phase."]}
    (ROOT/"data_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    tests=[]
    # Structural ablations
    tests += [
        Params("full_stack_vcp",full_stack=True),
        Params("early_vcp_no_accel",rsacc=-100),
        Params("early_vcp",rsacc=10),
        Params("early_no_vcp",rsacc=10,use_vcp=False),
        Params("early_vcp_breadth45",rsacc=10,breadth=0.45),
        Params("early_vcp_breadth50",rsacc=10,breadth=0.50),
        Params("early_vcp_breadth55",rsacc=10,breadth=0.55),
    ]
    # Sensitivity / frontier candidates
    for rs in [85,90,95]:
        for ratio in [0.95,0.97,0.99]:
            tests.append(Params(f"grid_rs{rs}_ma{ratio:.2f}",rs=rs,ma_ratio=ratio))
    for stop in [0.05,0.06,0.07,0.08]:
        tests.append(Params(f"grid_stop{int(stop*100)}",stop=stop))
    for bv in [1.0,1.2,1.3,1.5]:
        tests.append(Params(f"grid_breakvol{bv:.1f}",breakvol=bv))
    for vr in [0.70,0.80,0.90]:
        tests.append(Params(f"grid_voldry{vr:.2f}",vol_dry=vr))
    # Deduplicate by name
    uniq={p.name:p for p in tests}
    tests=list(uniq.values())

    summaries=[]; curves={}; tradebooks={}
    for i,p in enumerate(tests,1):
        print(f"[{i}/{len(tests)}] {p.name}",flush=True)
        c,t=simulate(df,p)
        m=metrics(c,t); m["Strategy"]=p.name
        m.update({"rs":p.rs,"rsacc":p.rsacc,"ma_ratio":p.ma_ratio,"last_range":p.last_range,"vol_dry":p.vol_dry,"atr_ratio":p.atr_ratio,"breakvol":p.breakvol,"stop":p.stop,"breadth":p.breadth,"full_stack":p.full_stack,"use_vcp":p.use_vcp})
        summaries.append(m); curves[p.name]=c; tradebooks[p.name]=t
    summary=pd.DataFrame(summaries).sort_values(["Calmar","CAGR"],ascending=False)
    summary.to_csv(RESULTS/"summary.csv",index=False)

    # Choose a robust candidate: among strategies with >=20 trades, maximize Calmar with CAGR tie-break.
    eligible=summary[summary.Trades>=20]
    best=(eligible if len(eligible) else summary).iloc[0]
    bname=best.Strategy
    bc=curves[bname]; bt=tradebooks[bname]
    yearly(bc).rename("return").to_csv(RESULTS/"yearly_best.csv")
    bc.to_csv(RESULTS/"equity_best.csv")
    bt.sort_values("ret",ascending=False).to_csv(RESULTS/"trades_best.csv",index=False)

    # top drawdown episodes (local troughs summarized simply)
    dd=bc.equity/bc.equity.cummax()-1
    worst=dd.nsmallest(10).rename("drawdown")
    worst.to_csv(RESULTS/"worst_drawdown_days.csv")

    report=[]
    report.append("# Minervini Korea — Price-only Phase Backtest\n")
    report.append(f"Data: {manifest['start']} to {manifest['end']}, {manifest['rows']:,} ticker-days, {manifest['tickers']:,} tickers.\n")
    report.append("## Important scope\nThis phase tests the price/liquidity/RS/RS-acceleration/Early-Trend/VCP/Pivot/long-hold engine on a point-in-time-like historical panel. It does **not** yet claim to test the requested era-sector and point-in-time earnings filters because historical sector membership and filing-timestamp fundamentals need a separate PIT layer. Therefore these results are a lower-layer validation, not the final Era-Leader strategy claim.\n")
    report.append("## Best robust candidate by Calmar (>=20 trades)\n")
    for k in ["Strategy","CAGR","MDD","Sharpe","Calmar","Final","Trades","WinRate","AvgTrade","MedianTrade","AvgHoldDays","ProfitFactor"]:
        v=best[k]
        if isinstance(v,(float,np.floating)):
            report.append(f"- {k}: {v:.4f}")
        else: report.append(f"- {k}: {v}")
    report.append("\n## Top strategies\n")
    report.append(summary.head(12).to_markdown(index=False))
    report.append("\n\n## Annual returns — selected candidate\n")
    yr=yearly(bc)
    report.append("\n".join([f"- {idx.year}: {val:.2%}" for idx,val in yr.items()]))
    report.append("\n\n## Audit notes\n- Signals use D close and entries use D+1 open.\n- 20 bps one-way slippage, 1.5 bps one-way commission, and 18 bps sell tax are included as a constant-cost stress assumption. Historical Korean tax schedules are not yet time-varying in this phase.\n- Delisted rows remain in marcap; a common-stock approximation is applied by market/code/name.\n- Corporate actions are normalized using KRX daily change ratios to create a continuous synthetic price series.\n- Historical sector membership and filing-time financials remain the principal missing inputs before the full Era-Leader claim can be audited.\n")
    (RESULTS/"report.md").write_text("\n".join(report),encoding="utf-8")
    print(summary.head(10).to_string(index=False),flush=True)
    print("BEST",bname,flush=True)

if __name__=="__main__":
    main()
