#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股ETF 每日跟踪 / TOP5 信号 / 回测系统 v1
数据源：东方财富公开行情接口，无需 API Key
用法：python3 etf_system.py [--out-dir output] [--skip-if-weekend]
输出：output/latest.json, output/dashboard.html, data/*.csv
"""
import argparse, base64, io, json, math, os, sys, time
from datetime import date, timedelta
import requests
import pandas as pd
import numpy as np

# 图表：matplotlib + Noto Sans CJK SC，保证中文不乱码
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        if os.path.exists(_fp):
            try:
                font_manager.fontManager.addfont(_fp)
            except Exception:
                pass
    # TTC 文件在 matplotlib 里注册为 JP 子族，但汉字字形通用，可正常显示中文
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except Exception:
    HAS_MPL = False

NAVY = "#0F2040"
GOLD = "#C9A86A"
GREY = "#8A94A6"
RED_UP = "#D92D20"
GREEN_DN = "#067647"

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "output")

# 名称以东方财富返回的实际名称为准（2026-10-30 实测）
POOL = [
    ("1.510300", "510300", "沪深300ETF华泰柏瑞"),
    ("0.159919", "159919", "沪深300ETF嘉实"),
    ("1.510500", "510500", "中证500ETF南方"),
    ("0.159982", "159982", "中证500ETF鹏华"),
    ("1.512100", "512100", "中证1000ETF南方"),
    ("0.159915", "159915", "创业板ETF易方达"),
    ("0.159783", "159783", "科创创业50ETF华夏"),
    ("1.512880", "512880", "证券ETF国泰"),
    ("1.512000", "512000", "券商ETF华宝"),
    ("1.510880", "510880", "红利ETF华泰柏瑞"),
    ("1.518880", "518880", "黄金ETF华安"),
    ("1.511010", "511010", "国债ETF国泰"),
    ("1.513100", "513100", "纳指ETF国泰"),
    ("1.513500", "513500", "标普500ETF博时"),
    ("1.513050", "513050", "中概互联网ETF易方达"),
    ("1.515050", "515050", "通信ETF华夏"),
    ("1.512010", "512010", "医药ETF易方达"),
    ("0.159928", "159928", "消费ETF汇添富"),
    ("1.516160", "516160", "新能源ETF南方"),
    ("0.159870", "159870", "化工ETF鹏华"),
]
BENCHMARK_CODE = "510300"
# 指数基准（指数本身，非ETF价格；不参与评分/TOP5，仅作累计收益对比）
INDEX_BENCHMARKS = [
    ("1.000001", "000001", "上证指数"),
    ("1.000300", "000300", "沪深300指数"),
]
COST = 0.0003  # 单边调仓成本（按换仓比例计）

def fetch_one(session, secid):
    today = date.today()
    beg = (today - timedelta(days=1100)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {"secid": secid, "klt": "101", "fqt": "1", "beg": beg, "end": end,
              "fields1": "f1,f2,f3,f4,f5", "fields2": "f51,f52,f53,f54,f55,f56,f57"}
    last_err = None
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=25)
            j = r.json()
            d = j.get("data") or {}
            klines = d.get("klines") or []
            if not klines:
                raise ValueError(f"empty klines for {secid}: {j}")
            rows = []
            for k in klines:
                p = k.split(",")
                rows.append({"date": p[0], "open": float(p[1]), "close": float(p[2]),
                             "high": float(p[3]), "low": float(p[4]),
                             "volume": float(p[5]), "amount": float(p[6])})
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df, d.get("name", "")
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed {secid}: {last_err}")

def rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ret5"] = df["close"] / df["close"].shift(5) - 1
    df["ret20"] = df["close"] / df["close"].shift(20) - 1
    df["ret60"] = df["close"] / df["close"].shift(60) - 1
    df["rsi"] = rsi_series(df["close"])
    df["vol20"] = df["ret1"].rolling(20).std()
    df["vol_ratio"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    # 60日最大回撤（截至当日）
    roll_max = df["close"].rolling(60).max()
    df["dd60"] = df["close"] / roll_max - 1
    return df

def score_row(row, pct_ret20, pct_ret60, pct_vol):
    trend = (15 if row["close"] > row["ma20"] else 0) + (15 if row["ma5"] > row["ma20"] else 0)
    momentum = 15 * pct_ret20 + 15 * pct_ret60
    rsi = row["rsi"]
    if pd.isna(rsi):
        rsi_score = 5
    elif 50 <= rsi <= 70:
        rsi_score = 10
    elif 40 <= rsi < 50 or 70 < rsi <= 80:
        rsi_score = 6
    else:
        rsi_score = 2
    vr = row["vol_ratio"]
    if pd.isna(vr):
        vol_score = 5
    elif vr >= 1.5:
        vol_score = 10
    elif vr <= 0.8:
        vol_score = 0
    else:
        vol_score = 10 * (vr - 0.8) / 0.7
    stability = 20 * (1 - pct_vol) if not pd.isna(pct_vol) else 10
    return float(trend + momentum + rsi_score + vol_score + stability)

def signal_from(score, row):
    if score >= 70 and row["close"] > row["ma20"] and (pd.isna(row["rsi"]) or row["rsi"] < 75):
        return "买入"
    if score < 40 or (row["close"] < row["ma20"] and row["ret20"] < 0) or (not pd.isna(row["rsi"]) and row["rsi"] > 80):
        return "卖出/规避"
    return "持有/观望"

def pct_rank(series):
    # 0-1 排名，NaN 排中间 0.5
    r = series.rank(pct=True)
    return r.fillna(0.5)

def compute_scores_on_day(frames, idx):
    """frames: dict code->df with indicators. idx: 行索引（各标的按日期对齐后使用共同日期轴时需注意）。
    本实现要求所有 df 已按共同日期重建索引（见 main），因此 idx 对应同一交易日。"""
    rows = {}
    for code, df in frames.items():
        if idx < len(df):
            rows[code] = df.iloc[idx]
    if not rows:
        return {}
    s_ret20 = pd.Series({c: r["ret20"] for c, r in rows.items()})
    s_ret60 = pd.Series({c: r["ret60"] for c, r in rows.items()})
    s_vol = pd.Series({c: r["vol20"] for c, r in rows.items()})
    pr20, pr60, pvol = pct_rank(s_ret20), pct_rank(s_ret60), pct_rank(s_vol)
    out = {}
    for code, row in rows.items():
        if pd.isna(row["ma20"]) or pd.isna(row["ret20"]):
            continue
        sc = score_row(row, pr20[code], pr60[code], pvol[code])
        out[code] = {"score": sc, "signal": signal_from(sc, row), "row": row}
    return out

def backtest(frames, common_dates, bench_code):
    # 回测最近 252 个交易日（信号日），持仓收益计入下一交易日
    n = len(common_dates)
    start = max(70, n - 253)  # 留足指标窗口，且约一年
    if n - start < 30:
        return None
    strat_rets, bench_rets, holdings_hist = [], [], []
    prev_hold = set()
    for i in range(start, n - 1):
        scores = compute_scores_on_day(frames, i)
        if len(scores) < 5:
            continue
        ranked_bt = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        buys_bt = [x for x in ranked_bt if x[1]["signal"] == "买入"]
        holds_bt = [x for x in ranked_bt if x[1]["signal"] == "持有/观望"]
        top5_bt = (buys_bt + holds_bt)[:5]  # 卖出/规避永不入选；不足5只时持有少于5只
        hold = set(c for c, _ in top5_bt)
        # 下一日收益
        rets = []
        for c in hold:
            df = frames[c]
            if i + 1 < len(df) and not pd.isna(df.iloc[i+1]["close"]) and not pd.isna(df.iloc[i]["close"]):
                rets.append(df.iloc[i+1]["close"] / df.iloc[i]["close"] - 1)
        if not rets:
            continue
        turnover = len(hold - prev_hold) / 5.0 if prev_hold else 1.0
        # 首次建仓也计成本；换仓部分计单边成本
        strat_rets.append(float(np.mean(rets)) - turnover * COST)
        # 基准
        bdf = frames[bench_code]
        bench_rets.append(float(bdf.iloc[i+1]["close"] / bdf.iloc[i]["close"] - 1))
        holdings_hist.append({"date": str(common_dates[i].date()), "top5": list(hold)})
        prev_hold = hold
    if len(strat_rets) < 30:
        return None
    s = np.array(strat_rets); b = np.array(bench_rets)
    def cum(x): return (1 + x).cumprod()
    sc, bc = cum(s), cum(b)
    def max_dd(cum_series):
        peak = np.maximum.accumulate(cum_series)
        return float(np.min(cum_series / peak - 1))
    def sharpe(x):
        return float((np.mean(x) * 252 - 0.02) / (np.std(x, ddof=1) * math.sqrt(252))) if np.std(x, ddof=1) > 0 else 0.0
    win = float(np.mean(s > 0))
    return {
        "period_start": str(common_dates[start].date()),
        "period_end": str(common_dates[n-2].date()) if n >= 2 else str(common_dates[-1].date()),
        "trading_days": int(len(s)),
        "strategy_cum": float(sc[-1] - 1),
        "benchmark_cum": float(bc[-1] - 1),
        "strategy_annualized": float((sc[-1] ** (252 / len(s)) - 1)) if sc[-1] > 0 else float(sc[-1] - 1),
        "benchmark_annualized": float((bc[-1] ** (252 / len(b)) - 1)) if bc[-1] > 0 else float(bc[-1] - 1),
        "strategy_max_dd": max_dd(sc),
        "benchmark_max_dd": max_dd(bc),
        "strategy_sharpe": sharpe(s),
        "benchmark_sharpe": sharpe(b),
        "win_rate": win,
        "strategy_curve": [round(float(x * 100), 2) for x in sc],
        "benchmark_curve": [round(float(x * 100), 2) for x in bc],
        "curve_dates": [str(common_dates[start + k + 1].date()) for k in range(len(s))],
    }

def _png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def _style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=12)
    ax.grid(axis="y", color="#E6E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#D0D5DD")
    ax.tick_params(labelsize=9, colors="#475467")

def make_charts(bt, top5, all_rows=None, bench_curves=None, sim=None, my=None):
    """返回 {net, drawdown, monthly, top5, daily, cumulative, bench_cmp, excess} 的 base64 PNG；无数据时为空。"""
    if not HAS_MPL or not bt:
        return {}
    import pandas as pd
    dates = pd.to_datetime(bt["curve_dates"])
    strat = np.array(bt["strategy_curve"], dtype=float)
    bench = np.array(bt["benchmark_curve"], dtype=float)
    out = {}
    # 1. 回测净值
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(dates, strat, color=NAVY, linewidth=2.2, label=f"TOP5策略（终值 {strat[-1]:.1f}）")
    ax.plot(dates, bench, color=GREY, linewidth=1.8, label=f"沪深300ETF基准（终值 {bench[-1]:.1f}）")
    ax.axhline(100, color=GOLD, linewidth=1.2, linestyle="--")
    ax.fill_between(dates, strat, 100, where=(strat >= 100), color=NAVY, alpha=0.06)
    _style_ax(ax, "回测净值走势（起点 = 100）")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.autofmt_xdate(rotation=0)
    out["net"] = _png_b64(fig)
    # 2. 回撤
    def dd(arr):
        peak = np.maximum.accumulate(arr)
        return arr / peak * 100 - 100
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(dates, dd(strat), 0, color=NAVY, alpha=0.18)
    ax.plot(dates, dd(strat), color=NAVY, linewidth=2, label=f"TOP5策略（最大回撤 {dd(strat).min():.2f}%）")
    ax.plot(dates, dd(bench), color=GREY, linewidth=1.6, label=f"基准（最大回撤 {dd(bench).min():.2f}%）")
    _style_ax(ax, "回撤走势（%）")
    ax.legend(frameon=False, fontsize=9, loc="best")
    out["drawdown"] = _png_b64(fig)
    # 3. 月度收益（由净值曲线按月末推导）
    df = pd.DataFrame({"date": dates, "s": strat, "b": bench})
    df["month"] = df["date"].dt.strftime("%Y-%m")
    last = df.groupby("month").last()
    prev_s, prev_b = 100.0, 100.0
    months, ms, mb = [], [], []
    for m, row in last.iterrows():
        months.append(m)
        ms.append(row["s"] / prev_s * 100 - 100)
        mb.append(row["b"] / prev_b * 100 - 100)
        prev_s, prev_b = row["s"], row["b"]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    w = 0.38
    ax.bar(x - w/2, ms, width=w, color=RED_UP, label="TOP5策略")
    ax.bar(x + w/2, mb, width=w, color="#2E7DD1", label="沪深300ETF基准")
    ax.axhline(0, color="#344054", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=35, ha="right", fontsize=8)
    _style_ax(ax, "月度收益对比（%）")
    ax.legend(frameon=False, fontsize=9, loc="best")
    out["monthly"] = _png_b64(fig)
    # 4. TOP5 评分条形图
    if top5:
        labels = [f"{t['code']} {t['name']}" for t in reversed(top5)]
        scores = [t["score"] for t in reversed(top5)]
        colors = [NAVY if t["signal"] == "买入" else GOLD for t in reversed(top5)]
        fig, ax = plt.subplots(figsize=(10, 3.0))
        bars = ax.barh(labels, scores, color=colors, height=0.55)
        ax.set_xlim(0, 100)
        ax.axvline(70, color=RED_UP, linewidth=1.2, linestyle="--")
        for bar, v in zip(bars, scores):
            ax.text(v + 1.5, bar.get_y() + bar.get_height()/2, f"{v:.1f}分", va="center", fontsize=10, fontweight="bold", color=NAVY)
        _style_ax(ax, "TOP5 综合评分（虚线 = 70 分买入线）")
        out["top5"] = _png_b64(fig)
    # 5. 当日盈亏分布（1日涨幅，红涨绿跌）与累计盈亏分布（60日涨幅）
    if all_rows:
        def _dist(rows, key, title):
            valid = [r for r in rows if r.get(key) is not None]
            valid = sorted(valid, key=lambda r: r[key])
            labels = [f"{r['code']}" for r in valid]
            vals = [r[key] * 100 for r in valid]
            colors = [RED_UP if v >= 0 else GREEN_DN for v in vals]
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.barh(labels, vals, color=colors, height=0.6)
            ax.axvline(0, color="#344054", linewidth=1)
            for i, v in enumerate(vals):
                ax.text(v + (0.15 if v >= 0 else -0.15), i, f"{v:+.2f}%", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7.5, color="#1A2332")
            _style_ax(ax, title)
            ax.tick_params(labelsize=8)
            return _png_b64(fig)
        out["daily"] = _dist(all_rows, "ret1", "当日涨幅分布（%）")
        out["cumulative"] = _dist(all_rows, "ret60", "60日累计涨幅分布（%）")
    # 6. 累计收益率走势对比：策略 vs 上证指数、沪深300指数（指数本身）
    if bench_curves:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        strat_pct = strat - 100
        ax.plot(dates, strat_pct, color=RED_UP, linewidth=2.2, label=f"TOP5策略（{strat_pct[-1]:+.2f}%）")
        _cols = {"000001": "#2E5AAC", "000300": "#E67E22"}
        _names = {"000001": "上证指数", "000300": "沪深300指数"}
        for code, vals in bench_curves.items():
            if len(vals) == len(dates):
                ax.plot(dates, vals, color=_cols.get(code, GREY), linewidth=1.6,
                        label=f"{_names.get(code, code)}（{vals[-1]:+.2f}%）")
        ax.axhline(0, color="#344054", linewidth=1)
        _style_ax(ax, "累计收益率走势对比（%）")
        ax.legend(frameon=False, fontsize=9, loc="best", ncol=2)
        out["bench_cmp"] = _png_b64(fig)
    # 7b. 模拟账户总资产走势
    if sim:
        sdates = pd.to_datetime(sim["curve_dates"])
        sassets = np.array(sim["curve_assets"], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(sdates, sassets, color=NAVY, linewidth=2.2, label=f"模拟账户总资产（终值 {sassets[-1]:,.0f}元）")
        ax.axhline(sim["initial"], color=GOLD, linewidth=1.2, linestyle="--", label=f"初始资金 {sim['initial']:,.0f}元")
        ax.fill_between(sdates, sassets, sim["initial"], where=(sassets >= sim["initial"]), color=NAVY, alpha=0.06)
        _style_ax(ax, "模拟账户总资产走势（元）")
        ax.legend(frameon=False, fontsize=9, loc="best")
        out["sim_assets"] = _png_b64(fig)
    # 7a. 我的持仓：归一化收益走势与单只盈亏率
    if my and my.get("rows"):
        mdates = pd.to_datetime(my["curve_dates"])
        mvals = np.array(my["curve_values"], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(mdates, mvals, color=NAVY, linewidth=2.4, label=f"我的持仓组合（终值 {mvals[-1]:.1f}）")
        _cols = ["#2E5AAC", "#E67E22", "#067647", "#7C3AED", "#B54708"]
        for i, r in enumerate(my["rows"]):
            rd = pd.to_datetime(r["curve_dates"])
            rv = np.array(r["curve_norm"], dtype=float)
            ax.plot(rd, rv, color=_cols[i % len(_cols)], linewidth=1.4, label=f"{r['code']} {r['name']}（{r['pnl_pct']*100:+.2f}%）")
        ax.axhline(100, color=GOLD, linewidth=1.2, linestyle="--")
        _style_ax(ax, "我的持仓收益走势（买入价 = 100，组合按成本加权）")
        ax.legend(frameon=False, fontsize=9, loc="best")
        out["my_curve"] = _png_b64(fig)
        labels = [f"{r['code']}" for r in my["rows"]]
        vals = [r["pnl_pct"] * 100 for r in my["rows"]]
        fig, ax = plt.subplots(figsize=(10, 3.0))
        ax.barh(labels, vals, color=[RED_UP if v >= 0 else GREEN_DN for v in vals], height=0.55)
        ax.axvline(0, color="#344054", linewidth=1)
        for i, v in enumerate(vals):
            ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold", color=NAVY)
        _style_ax(ax, "我的持仓单只盈亏率（买入以来，%）")
        out["my_pnl"] = _png_b64(fig)
    # 7. 超额收益曲线：策略净值 - 沪深300ETF净值（百分点，红正蓝负）
    excess = strat - bench
    fig, ax = plt.subplots(figsize=(10, 3.4))
    cols = [RED_UP if v >= 0 else "#2E7DD1" for v in excess]
    ax.bar(dates, excess, width=2.5, color=cols)
    ax.axhline(0, color="#344054", linewidth=1)
    _style_ax(ax, "超额收益曲线（策略 - 沪深300ETF，百分点）")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=RED_UP, label="超额为正：策略跑赢基准"), Patch(facecolor="#2E7DD1", label="超额为负：策略跑输基准")], frameon=False, fontsize=9, loc="best")
    out["excess"] = _png_b64(fig)
    return out


def compute_my_holdings(session):
    """我的持仓：读脚本目录下的 holdings.json，逐只抓行情，算买入以来收益与组合曲线。
    口径：单只盈亏率 = 最新价/成本价 - 1；组合曲线按成本（港币按 hkd_cny 折人民币）加权，
    买入前该只按成本价计（视为现金），买入后跟随市价；缺失且无缓存的标的剔除并在 missing 中注明，不编造价格。"""
    import json as _json
    path = os.path.join(BASE, "holdings.json")
    if not os.path.exists(path):
        return None
    try:
        cfg = _json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"MY HOLDINGS config fail: {e}", file=sys.stderr)
        return None
    entries = cfg.get("holdings") or []
    if not entries:
        return None
    rate = float(cfg.get("hkd_cny") or 0.92)
    rows, missing, cached = [], [], []
    for e in entries:
        code = str(e.get("code", "")).strip()
        market = str(e.get("market", "A股")).strip()
        try:
            shares = float(e.get("shares"))
            buy_price = float(e.get("buy_price"))
            buy_date = pd.to_datetime(e.get("buy_date")).date()
        except Exception:
            missing.append({"code": code, "label": market, "error": "holdings.json 字段不完整"})
            continue
        if market == "港股通":
            secid = f"116.{code.zfill(5)}"
            currency = "HKD"
        else:
            secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
            currency = "CNY"
        cache_path = os.path.join(DATA_DIR, f"hold_{code}.csv")
        try:
            df, real_name = fetch_one(session, secid)
            df.to_csv(cache_path, index=False)
            print(f"MY OK {code} {real_name} rows={len(df)} last={df.iloc[-1]['date'].date()} close={df.iloc[-1]['close']}")
        except Exception as ex:
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    real_name = e.get("name", code)
                    cached.append({"code": code, "label": market, "last": str(df.iloc[-1]['date'].date())})
                    print(f"MY CACHED {code} rows={len(df)} last={df.iloc[-1]['date'].date()}")
                except Exception as ce:
                    missing.append({"code": code, "label": market, "error": str(ce)})
                    continue
            else:
                missing.append({"code": code, "label": market, "error": str(ex)})
                print(f"MY FAIL {code} {ex}", file=sys.stderr)
                continue
            time.sleep(0.4)
        else:
            time.sleep(0.4)
        df = df.sort_values("date").reset_index(drop=True)
        latest = df.iloc[-1]
        price = float(latest["close"])
        value = shares * price
        cost = shares * buy_price
        # 组合曲线用：从最早买入日到最新日的日历轴，买入前按成本价计
        sub = df[df["date"].dt.date >= buy_date]
        if sub.empty:
            sub = df.iloc[-1:]
        rows.append({
            "code": code, "market": market, "currency": currency, "name": real_name or code,
            "buy_date": str(buy_date), "shares": shares, "buy_price": buy_price,
            "price": price, "latest_date": str(latest["date"].date()),
            "value": value, "cost": cost, "pnl": value - cost,
            "pnl_pct": (price / buy_price - 1) if buy_price else 0.0,
            "days": (latest["date"].date() - buy_date).days,
            "curve_dates": [str(d.date()) for d in sub["date"]],
            "curve_norm": [float(c / buy_price * 100) for c in sub["close"]],
        })
    if not rows:
        return {"simulated": bool(cfg.get("simulated")), "rows": [], "missing": missing, "cached": cached}
    total_cost_cny = sum(r["cost"] * (rate if r["currency"] == "HKD" else 1.0) for r in rows)
    total_value_cny = sum(r["value"] * (rate if r["currency"] == "HKD" else 1.0) for r in rows)
    for r in rows:
        r["weight"] = (r["cost"] * (rate if r["currency"] == "HKD" else 1.0) / total_cost_cny) if total_cost_cny else 0.0
    # 组合曲线：日历轴为最早买入日到最新日，买入前按 100 计
    all_dates = sorted(set(d for r in rows for d in r["curve_dates"]))
    # 用每只的曲线做前向填充，得到组合加权净值
    curve_vals = []
    for d in all_dates:
        tot = 0.0
        for r in rows:
            # 找到该日或之前最近的归一化值，买入前为 100
            vals = [(cd, cv) for cd, cv in zip(r["curve_dates"], r["curve_norm"]) if cd <= d]
            v = vals[-1][1] if vals else 100.0
            tot += r["weight"] * v
        curve_vals.append(round(float(tot), 2))
    peak = np.maximum.accumulate(np.array(curve_vals, dtype=float))
    max_dd = float((np.array(curve_vals) / peak * 100 - 100).min()) if curve_vals else 0.0
    return {
        "simulated": bool(cfg.get("simulated")), "hkd_cny": rate, "rows": rows,
        "missing": missing, "cached": cached, "curve_dates": all_dates, "curve_values": curve_vals,
        "total_cost_cny": total_cost_cny, "total_value_cny": total_value_cny,
        "total_pnl_cny": total_value_cny - total_cost_cny,
        "total_pnl_pct": (total_value_cny / total_cost_cny - 1) if total_cost_cny else 0.0,
        "max_dd": max_dd / 100.0,
        "start_date": min(r["buy_date"] for r in rows), "end_date": rows[0]["latest_date"],
    }

def simulate_account(frames, common_dates, latest_idx, bt, initial=200000.0):
    """20万元模拟账户：按每日TOP5信号（买入优先、持有补足、卖出不入选）等权调仓。
    调仓在当日收盘价进行，成本按换仓比例计单边0.03%（与回测口径一致的简化模型）。
    模拟用分数份额，实际ETF交易需100份整数倍，这里不做取整，差额计入说明。
    返回最终总资产、累计/今日盈亏、持仓明细与总资产曲线。"""
    if not bt:
        return None
    try:
        start_idx = [i for i, d in enumerate(common_dates) if str(d.date()) == bt["period_start"]][0]
    except Exception:
        return None
    cash = float(initial)
    holdings = {}
    curve_dates, curve_assets = [], []
    prev_total_after = None
    daily_pnl, daily_pct = 0.0, 0.0
    for i in range(start_idx, latest_idx + 1):
        scores = compute_scores_on_day(frames, i)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top5 = [c for c, _ in ([x for x in ranked if x[1]["signal"] == "买入"] + [x for x in ranked if x[1]["signal"] == "持有/观望"])[:5]]
        if not top5:
            continue
        date_i = common_dates[i].date()
        total_before = cash
        for code, h in holdings.items():
            if i < len(frames[code]) and not pd.isna(frames[code].iloc[i]["close"]):
                total_before += h["shares"] * float(frames[code].iloc[i]["close"])
        if prev_total_after is not None:
            daily_pnl = total_before - prev_total_after
            daily_pct = daily_pnl / prev_total_after if prev_total_after else 0.0
        if holdings:
            changed = len(set(top5) - set(holdings.keys()))
            cost = total_before * (changed / len(top5)) * COST
        else:
            cost = total_before * COST
        total_after = total_before - cost
        target_per = total_after / len(top5)
        new_holdings = {}
        for code in top5:
            price = float(frames[code].iloc[i]["close"])
            target_shares = target_per / price if price else 0.0
            if code in holdings and holdings[code]["shares"] > 0:
                old = holdings[code]
                if target_shares > old["shares"]:
                    new_cost = (old["shares"] * old["cost"] + (target_shares - old["shares"]) * price) / target_shares if target_shares else price
                else:
                    new_cost = old["cost"]
                entry = old["entry_date"]
            else:
                new_cost = price
                entry = date_i
            new_holdings[code] = {"shares": float(target_shares), "cost": float(new_cost), "entry_date": entry, "price": float(price)}
        holdings = new_holdings
        cash = total_after - sum(target_per for _ in top5)
        curve_dates.append(str(date_i))
        curve_assets.append(float(total_after))
        prev_total_after = total_after
    if not holdings or not curve_assets:
        return None
    final_total = float(curve_assets[-1])
    hold_list = []
    for code, h in holdings.items():
        value = h["shares"] * h["price"]
        pnl = (h["price"] - h["cost"]) * h["shares"]
        hold_list.append({
            "code": code, "shares": round(h["shares"], 2), "cost": round(h["cost"], 3),
            "price": round(h["price"], 3), "value": round(value, 2), "pnl": round(pnl, 2),
            "pnl_pct": float(h["price"] / h["cost"] - 1) if h["cost"] else 0.0,
            "weight": float(value / final_total) if final_total else 0.0,
            "days": int((common_dates[latest_idx].date() - h["entry_date"]).days),
            "entry_date": str(h["entry_date"]),
        })
    hold_list = sorted(hold_list, key=lambda x: x["value"], reverse=True)
    return {
        "initial": float(initial), "final_total": final_total,
        "holding_value": float(sum(h["value"] for h in hold_list)),
        "cash": float(final_total - sum(h["value"] for h in hold_list)),
        "cumulative_pnl": float(final_total - initial),
        "cumulative_pct": float(final_total / initial - 1),
        "daily_pnl": float(daily_pnl), "daily_pct": float(daily_pct),
        "holdings": hold_list, "curve_dates": curve_dates, "curve_assets": curve_assets,
        "start_date": str(common_dates[start_idx].date()), "end_date": str(common_dates[latest_idx].date()),
    }

def svg_curve(strat, bench):
    # 归一化到 100 起点，画两条折线
    W, H, PL, PR, PT, PB = 860, 280, 46, 16, 18, 30
    vals = strat + bench
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    def xy(i, v):
        x = PL + (W - PL - PR) * (i / max(1, len(vals)//2 - 1) if False else i / max(1, len(strat) - 1))
        y = PT + (H - PT - PB) * (1 - (v - lo) / (hi - lo))
        return x, y
    def path(arr):
        pts = [f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(arr)]
        return "M" + " L".join(pts)
    # y 轴刻度：最低、中间、最高
    ticks = ""
    for v, label in [(lo, f"{lo:.1f}"), ((lo+hi)/2, f"{(lo+hi)/2:.1f}"), (hi, f"{hi:.1f}")]:
        _, y = xy(0, v)
        ticks += f'<text x="{PL-6}" y="{y+4}" text-anchor="end" font-size="11" fill="#8a94a6">{label}</text><line x1="{PL}" y1="{y}" x2="{W-PR}" y2="{y}" stroke="#e6e9ef" stroke-width="1"/>'
    return f'''<svg viewBox="0 0 {W} {H}" width="100%" style="background:#fff;border:1px solid #e6e9ef;border-radius:10px">
    {ticks}
    <path d="{path(bench)}" fill="none" stroke="#8a94a6" stroke-width="2"/>
    <path d="{path(strat)}" fill="none" stroke="#0f2040" stroke-width="2.5"/>
    <text x="{PL}" y="{H-8}" font-size="11" fill="#8a94a6">回测净值（起点=100）</text>
    <text x="{W-PR}" y="{H-8}" text-anchor="end" font-size="11" fill="#8a94a6">深蓝=TOP5策略　灰=沪深300ETF基准</text>
    </svg>'''

def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:+.{digits}f}%"

def cls_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "up" if x > 0 else ("down" if x < 0 else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--skip-if-weekend", action="store_true")
    args = ap.parse_args()
    if args.skip_if_weekend and date.today().weekday() >= 5:
        print("weekend, skip")
        return 0
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    frames, meta, missing, cached = {}, {}, [], []
    for secid, code, label in POOL:
        try:
            df, real_name = fetch_one(session, secid)
            df = add_indicators(df)
            frames[code] = df
            meta[code] = {"secid": secid, "label": label, "real_name": real_name}
            df.to_csv(os.path.join(DATA_DIR, f"{code}.csv"), index=False)
            print(f"OK {code} {real_name} rows={len(df)} last={df.iloc[-1]['date'].date()} close={df.iloc[-1]['close']}")
        except Exception as e:
            # 抓取失败时用上次成功缓存兜底，避免单只 ETF 缺失导致全池排名与回测结果跳变
            cache_path = os.path.join(DATA_DIR, f"{code}.csv")
            if os.path.exists(cache_path):
                try:
                    cdf = pd.read_csv(cache_path, parse_dates=["date"])
                    cdf = add_indicators(cdf)
                    frames[code] = cdf
                    meta[code] = {"secid": secid, "label": label, "real_name": label}
                    cached.append({"code": code, "label": label, "last": str(cdf.iloc[-1]['date'].date())})
                    print(f"CACHED {code} rows={len(cdf)} last={cdf.iloc[-1]['date'].date()}")
                    continue
                except Exception as ce:
                    print(f"CACHE FAIL {code} {ce}", file=sys.stderr)
            missing.append({"code": code, "label": label, "error": str(e)})
            print(f"FAIL {code} {e}", file=sys.stderr)
        time.sleep(0.4)
    if len(frames) < 5:
        print("too few ETFs fetched", file=sys.stderr)
        return 2
    # 共同日期轴：取交集日期，保证同一 idx 对应同一交易日
    common = None
    for df in frames.values():
        s = set(df["date"])
        common = s if common is None else (common & s)
    common_dates = sorted(common)
    # 按共同日期重建索引
    aligned = {}
    for code, df in frames.items():
        d = df.set_index("date").reindex(common_dates)
        # 指标已在原 df 上算好，重建后保留；若某日缺失则该日不参与打分
        aligned[code] = d.reset_index()
    frames = aligned
    latest_idx = len(common_dates) - 1
    latest_date = common_dates[-1].date()
    scores = compute_scores_on_day(frames, latest_idx)
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    # TOP5：买入优先、持有补足，卖出/规避永不入选；不足5只时如实少于5只
    top5 = ([x for x in ranked if x[1]["signal"] == "买入"]
            + [x for x in ranked if x[1]["signal"] == "持有/观望"])[:5]
    # 情绪代理
    rets20 = [v["row"]["ret20"] for v in scores.values() if not pd.isna(v["row"]["ret20"])]
    breadth = float(np.mean([1 if x > 0 else 0 for x in rets20])) if rets20 else 0.5
    avg_ret20 = float(np.mean(rets20)) if rets20 else 0.0
    sentiment = 100 * (0.6 * breadth + 0.4 * max(0.0, min(1.0, 0.5 + avg_ret20 * 5)))
    sentiment_label = "偏多" if sentiment >= 60 else ("偏空" if sentiment < 40 else "中性")
    bt = backtest(frames, common_dates, BENCHMARK_CODE)
    # 指数基准：上证指数、沪深300指数（指数本身，非ETF价格），不参与评分/TOP5
    index_frames, index_missing = {}, []
    for secid, icode, iname in INDEX_BENCHMARKS:
        try:
            idf, ireal = fetch_one(session, secid)
            index_frames[icode] = idf
            idf.to_csv(os.path.join(DATA_DIR, f"index_{icode}.csv"), index=False)
            print(f"INDEX OK {icode} {ireal} rows={len(idf)} last={idf.iloc[-1]['date'].date()} close={idf.iloc[-1]['close']}")
        except Exception as e:
            cache_path = os.path.join(DATA_DIR, f"index_{icode}.csv")
            if os.path.exists(cache_path):
                try:
                    cdf = pd.read_csv(cache_path, parse_dates=["date"])
                    index_frames[icode] = cdf
                    print(f"INDEX CACHED {icode} rows={len(cdf)} last={cdf.iloc[-1]['date'].date()}")
                    continue
                except Exception as ce:
                    print(f"INDEX CACHE FAIL {icode} {ce}", file=sys.stderr)
            index_missing.append({"code": icode, "label": iname, "error": str(e)})
            print(f"INDEX FAIL {icode} {e}", file=sys.stderr)
        time.sleep(0.4)
    # 指数基准在回测期的累计收益曲线（与策略曲线日期对齐；曲线日缺失时用最近已有点位顺延）
    bench_curves, index_ret20 = {}, {}
    if bt:
        try:
            for secid, icode, iname in INDEX_BENCHMARKS:
                if icode not in index_frames:
                    continue
                df = index_frames[icode]
                cmap = {str(d.date()): float(c) for d, c in zip(df["date"], df["close"])}
                base = cmap.get(bt["period_start"])
                if base is None or base == 0:
                    continue
                vals, last_val = [], 0.0
                for ds in bt["curve_dates"]:
                    if ds in cmap:
                        last_val = float(cmap[ds] / base - 1) * 100
                    vals.append(last_val)
                bench_curves[icode] = vals
                if len(df) >= 21:
                    index_ret20[icode] = float(df.iloc[-1]["close"] / df.iloc[-21]["close"] - 1)
        except Exception as e:
            print(f"bench_curves skipped: {e}", file=sys.stderr)
    # 最新行数据
    def row_dict(code, info):
        r = info["row"]
        return {
            "code": code, "name": meta[code]["real_name"] or meta[code]["label"],
            "price": round(float(r["close"]), 3),
            "ret1": None if pd.isna(r["ret1"]) else float(r["ret1"]),
            "ret5": None if pd.isna(r["ret5"]) else float(r["ret5"]),
            "ret20": None if pd.isna(r["ret20"]) else float(r["ret20"]),
            "ret60": None if pd.isna(r["ret60"]) else float(r["ret60"]),
            "ma20": None if pd.isna(r["ma20"]) else round(float(r["ma20"]), 3),
            "rsi": None if pd.isna(r["rsi"]) else round(float(r["rsi"]), 1),
            "vol_ratio": None if pd.isna(r["vol_ratio"]) else round(float(r["vol_ratio"]), 2),
            "score": round(float(info["score"]), 1),
            "signal": info["signal"],
        }
    result = {
        "as_of": str(latest_date),
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "pool_size": len(frames),
        "missing": missing,
        "cached": cached,
        "sentiment": {"score": round(sentiment, 1), "label": sentiment_label,
                      "breadth": round(breadth, 3), "avg_ret20": avg_ret20},
        "top5": [row_dict(c, info) for c, info in top5],
        "all": [row_dict(c, info) for c, info in ranked],
        "sell_list": [row_dict(c, info) for c, info in ranked if info["signal"] == "卖出/规避"],
        "backtest": bt,
        "benchmark_curves": bench_curves,
        "index_benchmarks": [{"code": icode, "name": iname, "cum": (bench_curves[icode][-1] / 100 if icode in bench_curves and bench_curves[icode] else None), "ret20": index_ret20.get(icode)} for _, icode, iname in INDEX_BENCHMARKS],
        "index_missing": index_missing,
        "sim_account": simulate_account(frames, common_dates, latest_idx, bt, initial=200000.0),
        "my_holdings": compute_my_holdings(session),
    }
    with open(os.path.join(args.out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # HTML 看板
    top_rows = ""
    for i, t in enumerate(result["top5"], 1):
        sig_cls = "buy" if t["signal"] == "买入" else ("sell" if t["signal"] == "卖出/规避" else "hold")
        top_rows += f"""<tr><td>{i}</td><td><b>{t['code']}</b><br><span class="muted">{t['name']}</span></td>
        <td>{t['price']}</td><td class="{cls_pct(t['ret5'])}">{fmt_pct(t['ret5'])}</td>
        <td class="{cls_pct(t['ret20'])}">{fmt_pct(t['ret20'])}</td><td class="{cls_pct(t['ret60'])}">{fmt_pct(t['ret60'])}</td>
        <td>{t['rsi'] if t['rsi'] is not None else '—'}</td><td><b>{t['score']}</b></td>
        <td><span class="sig {sig_cls}">{t['signal']}</span></td></tr>"""
    all_rows = ""
    for t in result["all"]:
        sig_cls = "buy" if t["signal"] == "买入" else ("sell" if t["signal"] == "卖出/规避" else "hold")
        all_rows += f"""<tr><td><b>{t['code']}</b><br><span class="muted">{t['name']}</span></td>
        <td>{t['price']}</td><td class="{cls_pct(t['ret20'])}">{fmt_pct(t['ret20'])}</td>
        <td>{t['rsi'] if t['rsi'] is not None else '—'}</td><td>{t['vol_ratio'] if t['vol_ratio'] is not None else '—'}</td>
        <td><b>{t['score']}</b></td><td><span class="sig {sig_cls}">{t['signal']}</span></td></tr>"""
    sell_text = "、".join([f"{s['code']}（{s['score']}分）" for s in result["sell_list"]]) + ("）" if result["sell_list"] else "暂无")
    if result["sell_list"]:
        sell_text = "、".join([f"{s['code']}（{s['score']}分）" for s in result["sell_list"]])
    bt = result["backtest"]
    charts = make_charts(bt, result["top5"], all_rows=result["all"], bench_curves=result["benchmark_curves"], sim=result["sim_account"], my=result["my_holdings"])
    def _img(key, alt):
        return f'<div class="chart"><img src="data:image/png;base64,{charts[key]}" alt="{alt}"/></div>' if key in charts else ""
    if bt:
        bt_html = f"""
        <table class="kpi-table"><tr>
          <td><div class="kpi-label">策略累计收益</div><div class="kpi-value {cls_pct(bt['strategy_cum'])}">{fmt_pct(bt['strategy_cum'])}</div><div class="muted">基准 {fmt_pct(bt['benchmark_cum'])}</div></div>
          <td><div class="kpi-label">策略年化收益</div><div class="kpi-value {cls_pct(bt['strategy_annualized'])}">{fmt_pct(bt['strategy_annualized'])}</div><div class="muted">基准 {fmt_pct(bt['benchmark_annualized'])}</div></div>
          <td><div class="kpi-label">策略最大回撤</div><div class="kpi-value">{fmt_pct(bt['strategy_max_dd'])}</div><div class="muted">基准 {fmt_pct(bt['benchmark_max_dd'])}</div></div>
          <td><div class="kpi-label">夏普比率</div><div class="kpi-value">{bt['strategy_sharpe']:.2f}</div><div class="muted">基准 {bt['benchmark_sharpe']:.2f}</div></div>
          <td><div class="kpi-label">胜率（日收益&gt;0占比）</div><div class="kpi-value">{bt['win_rate']*100:.1f}%</div><div class="muted">{bt['trading_days']} 个交易日</div></div>
        </tr></table>
        <p class="muted">回测区间：{bt['period_start']} 至 {bt['period_end']}；每日等权持有 TOP5、每日调仓、单边成本 0.03%（按换仓比例计）；信号当日收盘计算，收益计入下一交易日，无未来函数。读图要点：净值图看长期跑赢幅度，回撤图看最难熬的阶段，月度图看胜负是否集中在某几个月。</p>
        {_img('net', '回测净值走势')}
        {_img('drawdown', '回撤走势')}
        {_img('monthly', '月度收益对比')}
        """
    else:
        bt_html = "<p>回测数据不足，暂未生成。</p>"
    missing_html = ""
    if cached:
        missing_html += "<p class='warn'>本次抓取失败、已用上次缓存兜底（价格为缓存日收盘价，非今日新数据）：" + "、".join([c["code"] + "（缓存至" + c["last"] + "）" for c in cached]) + "</p>"
    if missing:
        missing_html += "<p class='warn'>数据缺失（已剔除当日评选，不编造价格）：" + "、".join([m["code"] for m in missing]) + "</p>"
    if index_missing:
        missing_html += "<p class='warn'>指数基准数据缺失（累计对比中该指数暂不显示，不编造价格）：" + "、".join([m["code"] + " " + m["label"] for m in index_missing]) + "</p>"
    my = result["my_holdings"]
    my_html = ""
    if my and my.get("rows"):
        my_rows_html = "".join([f"<tr><td><b>{r['code']}</b></td><td>{r['name']}</td><td>{r['market']}</td><td>{r['buy_date']}</td><td>{r['shares']:,.0f}</td><td>{r['buy_price']}</td><td>{r['price']}</td><td>{r['value']:,.2f} {r['currency']}</td><td class='{cls_pct(r['pnl_pct'])}'>{r['pnl']:+,.2f} {r['currency']}</td><td class='{cls_pct(r['pnl_pct'])}'>{fmt_pct(r['pnl_pct'])}</td><td>{r['weight']*100:.1f}%</td><td>{r['days']}</td></tr>" for r in my["rows"]])
        my_warn = ""
        if my.get("simulated"):
            my_warn += "<p class='warn'>模拟数据演示，非真实持仓：下面是演示用的持仓，改仓库里的 holdings.json（代码、市场、买入日期、数量、买入价）后，Actions 下次运行会自动换成你的持仓。</p>"
        if my.get("cached"):
            my_warn += "<p class='warn'>本次抓取失败、已用上次缓存兜底：" + "、".join([c["code"] + "（缓存至" + c["last"] + "）" for c in my["cached"]]) + "</p>"
        if my.get("missing"):
            my_warn += "<p class='warn'>持仓数据缺失（已剔除，不编造价格）：" + "、".join([m["code"] for m in my["missing"]]) + "</p>"
        my_html = f"""<div class="card"><h2>我的持仓（买入以来）</h2>{my_warn}<p class="muted">口径：单只盈亏率 = 最新价 / 成本价 - 1；组合曲线按成本加权（港币按 {my['hkd_cny']} 人民币/港元折算，买入前按成本价计）；A股与港股交易日不同，曲线按日历对齐、缺失日用前一收盘价顺延。持仓期：{my['start_date']} 至 {my['end_date']}。</p>
    <div class="kpi-grid kpi-grid-4">
      <div class="kpi"><div class="kpi-label">总成本（折人民币）</div><div class="kpi-value">{my['total_cost_cny']:,.2f} 元</div><div class="muted">港币按 {my['hkd_cny']} 折算</div></div>
      <div class="kpi"><div class="kpi-label">总市值（折人民币）</div><div class="kpi-value">{my['total_value_cny']:,.2f} 元</div><div class="muted">按最新收盘价</div></div>
      <div class="kpi"><div class="kpi-label">累计盈亏（折人民币）</div><div class="kpi-value {cls_pct(my['total_pnl_pct'])}">{my['total_pnl_cny']:+,.2f} 元（{fmt_pct(my['total_pnl_pct'])}）</div><div class="muted">买入以来，未扣交易费用</div></div>
      <div class="kpi"><div class="kpi-label">组合最大回撤</div><div class="kpi-value">{fmt_pct(my['max_dd'])}</div><div class="muted">组合曲线最高点以来</div></div>
    </div>
    <div class="table-wrap"><table class="data"><tr><th>代码</th><th>名称</th><th>市场</th><th>买入日期</th><th>数量（股）</th><th>成本价</th><th>最新价</th><th>市值</th><th>盈亏额</th><th>盈亏率</th><th>仓位</th><th>天数</th></tr>{my_rows_html}</table></div>
    {_img('my_curve', '我的持仓收益走势')}{_img('my_pnl', '我的持仓单只盈亏率')}</div>"""
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>A股ETF每日看板 {result['as_of']}</title>
    <style>
    @page{{size:A4;margin:14mm}}
    *{{box-sizing:border-box}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#eef1f5;color:#1a2332;margin:0;line-height:1.55}}
    .wrap{{width:min(1120px,calc(100% - 32px));margin:0 auto;padding:24px 0 40px}}
    header{{width:100%;background:#0f2040;color:#fff;border-radius:14px;padding:26px 28px;border-bottom:4px solid #c9a86a}}
    header h1{{margin:0 0 8px;font-size:26px;line-height:1.25}} header p{{margin:4px 0;color:#cbd5e1}}
    .gold{{color:#c9a86a}} .card{{width:100%;background:#fff;border-radius:12px;padding:22px 24px;margin-top:16px;box-shadow:0 2px 10px rgba(15,32,64,.06);page-break-inside:avoid;overflow:hidden}}
    h2{{font-size:18px;margin:0 0 14px;border-left:4px solid #c9a86a;padding-left:10px}} .muted{{color:#667085;font-size:13px;line-height:1.6}}
    .kpi-grid{{display:grid;gap:12px;width:100%;margin-top:16px}} .kpi-grid-4{{grid-template-columns:repeat(4,1fr)}} .kpi-grid-3{{grid-template-columns:repeat(3,1fr)}} .kpi-grid-2{{grid-template-columns:repeat(2,1fr)}} .kpi-grid-5{{grid-template-columns:repeat(5,1fr)}}
    .kpi{{background:#f7f8fa;border:1px solid #eef1f5;border-radius:10px;padding:14px 16px;min-width:0}}
    .kpi-label{{font-size:13px;color:#667085}} .kpi-value{{font-size:20px;font-weight:700;margin:6px 0;line-height:1.3;overflow-wrap:anywhere}} .kpi .muted{{font-size:12.5px}}
    .kpi-table{{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:8px 0;margin:16px -8px 0}} .kpi-table td{{background:#f7f8fa;border-radius:10px;padding:14px;vertical-align:top}}
    .table-wrap{{width:100%;overflow-x:auto;margin-top:8px}} table.data{{width:100%;min-width:640px;border-collapse:collapse;font-size:14px}} table.data th{{text-align:left;color:#667085;border-bottom:2px solid #0f2040;padding:10px 10px;white-space:nowrap}} table.data td{{border-bottom:1px solid #eef1f5;padding:10px;vertical-align:top;white-space:nowrap}} table.data td:nth-child(2){{white-space:normal}}
    .chart{{width:100%;margin:12px 0 0}} .chart img,.chart svg{{width:100%;height:auto;display:block;background:#fff;border:1px solid #e6e9ef;border-radius:10px;padding:6px}}
    .split{{display:grid;grid-template-columns:1fr 1fr;gap:16px;width:100%;margin-top:12px}} .split .chart{{margin:0}}
    .up{{color:#d92d20;font-weight:600}} .down{{color:#067647;font-weight:600}}
    .sig{{display:inline-block;padding:3px 9px;border-radius:20px;font-size:13px;font-weight:700;white-space:nowrap}} .buy{{background:#fef3f2;color:#d92d20;border:1px solid #fecdca}} .sell{{background:#ecfdf3;color:#067647;border:1px solid #a6f4c5}} .hold{{background:#fffaeb;color:#b54708;border:1px solid #fedf89}}
    .warn{{background:#fffaeb;border:1px solid #fedf89;padding:10px 12px;border-radius:8px;margin-top:16px}} .foot{{font-size:12.5px;color:#667085;line-height:1.7}}
    .page-break{{page-break-before:always}} .chart-note{{font-size:13px;color:#475467;background:#f7f8fa;border-radius:8px;padding:10px 12px}}
    @media (max-width:720px){{.wrap{{width:calc(100% - 24px);padding-top:16px}} header{{padding:20px}} header h1{{font-size:22px}} .card{{padding:16px}} .kpi-grid-4,.kpi-grid-3,.kpi-grid-5,.kpi-grid-2{{grid-template-columns:1fr 1fr}} .split{{grid-template-columns:1fr}} .kpi-value{{font-size:18px}} table.data{{font-size:13px}}}}
    @media print{{body{{background:#fff}} .wrap{{width:100%}} .card{{box-shadow:none;border:1px solid #e6e9ef}}}}
    </style></head><body><div class="wrap">
    <header><div class="gold">ETF 组合监控中枢 · 每日跟踪与回测</div><h1>ETF 每日看板</h1>
    <p>数据截至：{result['as_of']} 收盘　｜　生成时间：{result['generated_at']}（北京时间）　｜　样本池：{result['pool_size']} 只 ETF　｜　下次更新：下一个交易日</p>
    <p>本期结论：市场情绪 <b>{result['sentiment']['label']}</b>（{result['sentiment']['score']}分），20日上涨家数占比 {result['sentiment']['breadth']*100:.0f}%，池内平均20日涨幅 {fmt_pct(result['sentiment']['avg_ret20'])}。</p></header>
    {missing_html}
    <div class="kpi-grid kpi-grid-4">
      <div class="kpi"><div class="kpi-label">策略累计收益（回测期）</div><div class="kpi-value {cls_pct(bt['strategy_cum']) if bt else ''}">{fmt_pct(bt['strategy_cum']) if bt else '—'}</div><div class="muted">回测 {bt['period_start']} 至 {bt['period_end'] if bt else ''}</div></div>
      <div class="kpi"><div class="kpi-label">相对沪深300ETF超额</div><div class="kpi-value {cls_pct((bt['strategy_cum']-bt['benchmark_cum']) if bt else 0)}">{fmt_pct((bt['strategy_cum']-bt['benchmark_cum']) if bt else 0)}</div><div class="muted">策略 - 基准</div></div>
      <div class="kpi"><div class="kpi-label">策略最大回撤</div><div class="kpi-value">{fmt_pct(bt['strategy_max_dd']) if bt else '—'}</div><div class="muted">基准 {fmt_pct(bt['benchmark_max_dd']) if bt else '—'}</div></div>
      <div class="kpi"><div class="kpi-label">夏普比率 / 胜率</div><div class="kpi-value">{bt['strategy_sharpe']:.2f} / {bt['win_rate']*100:.1f}%</div><div class="muted">基准夏普 {bt['benchmark_sharpe']:.2f}</div></div>
    </div>
    <div class="card"><p class="muted"><b>组合诊断：</b>市场情绪 {result['sentiment']['label']}（{result['sentiment']['score']}分）　｜　上涨家数占比 {result['sentiment']['breadth']*100:.0f}%　｜　TOP5持仓 {len(result['top5'])} 只，等权 {100/len(result['top5']):.0f}%/只　｜　TOP5平均评分 {sum(t['score'] for t in result['top5'])/len(result['top5']):.1f}分　｜　回测 {bt['trading_days'] if bt else '—'} 个交易日</p></div>
    <div class="card"><h2>模拟账户（初始 20 万元）</h2><p class="muted">按每日 TOP5 信号等权模拟调仓，收盘价交易，单边成本 0.03%（按换仓比例计），分数份额模拟、实际交易需 100 份整数倍。模拟期：{result['sim_account']['start_date'] if result['sim_account'] else '—'} 至 {result['sim_account']['end_date'] if result['sim_account'] else '—'}。</p>
    <div class="kpi-grid kpi-grid-4">
      <div class="kpi"><div class="kpi-label">总资产</div><div class="kpi-value">{result['sim_account']['final_total']:,.2f} 元</div><div class="muted">初始 200,000.00 元</div></div>
      <div class="kpi"><div class="kpi-label">持有市值</div><div class="kpi-value">{result['sim_account']['holding_value']:,.2f} 元</div><div class="muted">现金 {result['sim_account']['cash']:,.2f} 元，仓位约 100%</div></div>
      <div class="kpi"><div class="kpi-label">累计盈亏</div><div class="kpi-value {cls_pct(result['sim_account']['cumulative_pct'])}">{result['sim_account']['cumulative_pnl']:+,.2f} 元（{fmt_pct(result['sim_account']['cumulative_pct'])}）</div><div class="muted">模拟期累计</div></div>
      <div class="kpi"><div class="kpi-label">今日盈亏（最新交易日）</div><div class="kpi-value {cls_pct(result['sim_account']['daily_pct'])}">{result['sim_account']['daily_pnl']:+,.2f} 元（{fmt_pct(result['sim_account']['daily_pct'])}）</div><div class="muted">调仓前市值变动，未扣当日调仓成本</div></div>
    </div>
    <div class="table-wrap"><table class="data"><tr><th>代码</th><th>名称</th><th>数量（份）</th><th>成本价</th><th>最新价</th><th>市值</th><th>盈亏额</th><th>盈亏率</th><th>仓位</th><th>天数</th></tr>{''.join([f"<tr><td><b>{h['code']}</b></td><td>{next((t['name'] for t in result['all'] if t['code']==h['code']), h['code'])}</td><td>{h['shares']:,.2f}</td><td>{h['cost']}</td><td>{h['price']}</td><td>{h['value']:,.2f}</td><td class='{cls_pct(h['pnl_pct'])}'>{h['pnl']:+,.2f}</td><td class='{cls_pct(h['pnl_pct'])}'>{fmt_pct(h['pnl_pct'])}</td><td>{h['weight']*100:.1f}%</td><td>{h['days']}</td></tr>" for h in result['sim_account']['holdings']]) if result['sim_account'] else ''}</table></div>
    <p class="muted">注：持仓盈亏仅为当前持仓未实现部分；累计盈亏还包括历史已卖出持仓的实现盈亏与调仓成本，两者不等是正常的。</p>
    {_img('sim_assets', '模拟账户总资产走势')}</div>
    {my_html}
    <div class="card"><h2>持仓概览</h2><p class="muted">按综合评分排序；买入优先、持有补足，卖出/规避标的不入选；买入+持有不足5只时，TOP5如实少于5只。评分构成：趋势30 + 动量30 + 强度20 + 稳定20。</p>
    <div class="table-wrap"><table class="data"><tr><th>代码</th><th>名称</th><th>权重</th><th>收盘价</th><th>5日</th><th>20日</th><th>60日</th><th>RSI</th><th>评分</th><th>信号</th></tr>{''.join([f"<tr><td><b>{t['code']}</b></td><td>{t['name']}</td><td>{100/len(result['top5']):.0f}%</td><td>{t['price']}</td><td class='{cls_pct(t['ret5'])}'>{fmt_pct(t['ret5'])}</td><td class='{cls_pct(t['ret20'])}'>{fmt_pct(t['ret20'])}</td><td class='{cls_pct(t['ret60'])}'>{fmt_pct(t['ret60'])}</td><td>{t['rsi'] if t['rsi'] is not None else '—'}</td><td><b>{t['score']}</b></td><td><span class='sig {'buy' if t['signal']=='买入' else 'hold'}'>{t['signal']}</span></td></tr>" for t in result['top5']])}</table></div>
    {_img('top5', 'TOP5综合评分')}</div>
    <div class="card"><h2>组合净值走势</h2>{_img('net', '回测净值走势')}{_img('drawdown', '回撤走势')}</div>
    <div class="card"><h2>当日与累计盈亏分布</h2><p class="muted">左：当日涨幅（1日）；右：累计涨幅（60日）。红涨绿跌，数值越大条形越长。</p>
    <div class="split">{_img('daily', '当日涨幅分布')}{_img('cumulative', '累计涨幅分布')}</div></div>
    <div class="card page-break"><h2>vs 基准对比</h2>
    <p class="muted">对比基准为指数本身（非ETF价格）：上证指数（000001）、沪深300指数（000300）。顶部“相对沪深300ETF超额”仍以可交易的沪深300ETF（510300）为基准，两者口径不同，数值会有差异，属正常。</p>
    <div class="kpi-grid kpi-grid-2">
      {''.join([f"<div class='kpi'><div class='kpi-label'>vs {name}</div><div class='kpi-value {cls_pct((bt['strategy_cum']*100 - vals[-1])/100) if bt and vals else ''}'>超额 {fmt_pct((bt['strategy_cum'] - vals[-1]/100) if bt and vals else 0)}</div><div class='muted'>指数回测累计 {fmt_pct(vals[-1]/100) if vals else '—'}　｜　指数20日 {fmt_pct(next((b['ret20'] for b in result['index_benchmarks'] if b['code']==code), None))}</div></div>" for code, name, vals in [('000001','上证指数', result['benchmark_curves'].get('000001')), ('000300','沪深300指数', result['benchmark_curves'].get('000300'))]])}
    </div>
    <p class="muted">超额 = 策略回测累计 - 指数回测累计；指数回测累计以回测期初收盘价为基数计算，与策略曲线日期对齐。指数20日为指数收盘价近20个交易日的涨幅。</p>
    {_img('bench_cmp', '累计收益率走势对比')}
    {_img('excess', '超额收益曲线')}
    <p class="chart-note"><b>超额图怎么看：</b>红色柱子 = 超额为正，策略跑赢沪深300ETF；蓝色柱子 = 超额为负，策略跑输沪深300ETF；柱子越长，跑赢/跑输幅度越大。0线以上为跑赢、以下为跑输。注：这张图的基准是沪深300ETF（510300），不是沪深300指数。</p>
    {_img('monthly', '月度收益对比')}
    <p class="chart-note"><b>月度图颜色：</b>红色 = TOP5策略，蓝色 = 沪深300ETF基准；同一月份两根柱子一对比，就能直观看到当月谁赢、赢多少。</p></div>
    <div class="card"><h2>卖出 / 规避名单</h2><p>{sell_text if result['sell_list'] else '暂无卖出/规避信号标的。'}</p>
    <p class="muted">卖出/规避触发条件：评分 &lt; 40，或收盘价 &lt; MA20 且20日涨幅为负，或 RSI &gt; 80。</p></div>
    <div class="card"><h2>全池评分表（附录）</h2><div class="table-wrap"><table class="data"><tr><th>ETF</th><th>收盘价</th><th>20日</th><th>RSI</th><th>量比</th><th>评分</th><th>信号</th></tr>{all_rows}</table></div></div>
    <div class="card"><p class="foot"><b>方法与口径：</b>趋势看收盘价与 MA5/MA20 关系；动量看20日、60日涨幅在池内排名；强度看 RSI 与成交量放大；稳定看20日波动率排名。情绪为代理指标（上涨家数占比 + 成交量 + 动量），非研报级舆情 NLP，后续可升级接入新闻情感分析。<br>
    <b>数据源：</b>东方财富公开行情接口；数据缺失标的已剔除，不编造。<br>
    <b>风险提示：</b>本看板为量化分析与信息参考，不构成买入、卖出或持有任何证券的建议；历史回测不代表未来收益。</p></div>
    </div></body></html>"""
    with open(os.path.join(args.out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    # 自动生成 PDF（供 Drive 预览渲染用；LibreOffice 不可用时只保留 HTML，不中断）
    try:
        import shutil, subprocess
        if shutil.which("soffice"):
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", args.out_dir,
                            os.path.join(args.out_dir, "dashboard.html")],
                           capture_output=True, timeout=120, check=False)
            print(f"PDF: {os.path.join(args.out_dir, 'dashboard.pdf')} exists={os.path.exists(os.path.join(args.out_dir, 'dashboard.pdf'))}")
    except Exception as e:
        print(f"PDF generation skipped: {e}", file=sys.stderr)
    print(f"DONE as_of={result['as_of']} top5={[t['code'] for t in result['top5']]}")
    if bt:
        print(f"BACKTEST {bt['period_start']}..{bt['period_end']} strat={bt['strategy_cum']*100:.2f}% bench={bt['benchmark_cum']*100:.2f}% sharpe={bt['strategy_sharpe']:.2f} maxdd={bt['strategy_max_dd']*100:.2f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
