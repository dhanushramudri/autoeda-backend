"""
Forecasting solution. Zero configuration: finds the best time axis + revenue-like measure across all datasets,
aggregates it to a regular series, explores it, backtests several models with a rolling origin, forecasts the best one
with intervals, validates against a naive baseline, and exports the forecast.

Stages are top-level functions `stage_*(ctx) -> (result, artifacts)` (runnable in the process pool). Each result carries
a `page` spec (see pages.py) that the generic frontend renderer draws.
"""

from __future__ import annotations

import re
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from ..eda.ts_columns import detect_time_columns, detect_value_columns, parse_time_column
from . import pages as P
from .common import clean

warnings.filterwarnings("ignore")

FREQ = {
    "D": dict(period="D", m=7, h_bt=14, h=30, label="daily"),
    "W": dict(period="W", m=52, h_bt=13, h=26, label="weekly"),
    "M": dict(period="M", m=12, h_bt=6, h=12, label="monthly"),
    "Q": dict(period="Q", m=4, h_bt=2, h=4, label="quarterly"),
    "Y": dict(period="Y", m=1, h_bt=1, h=2, label="yearly"),
}
STRONG_PERIOD = re.compile(r"(period|month|snapshot|week|quarter)", re.I)
WEAK_PERIOD = re.compile(r"(date|day|renewal|time)", re.I)
BAD_TIME = re.compile(r"(regist|creat|birth|proforma|closed|paid|start|join)", re.I)
MODEL_NOTES = {
    "Naive": "repeats the last value", "Seasonal naive": "repeats the last full season", "Drift": "last value plus the average historical change",
    "Exponential smoothing": "level, damped trend and season (ETS)", "ARIMA": "autoregressive model (seasonal when there is enough history)",
    "Gradient boosting": "lagged values with a boosted-tree model",
}


def _rank_value(name: str) -> int:
    n = name.lower()
    if re.search(r"^(total_)?(amount|revenue)$|(^|_)(revenue|sales|arr|mrr|turnover|gmv)($|_)", n):
        return 3
    if re.search(r"amount|net|gross|price|value|spend|cost|volume|qty|quantity", n):
        return 2
    return 1


def _series_label(key: str) -> str:
    return FREQ[key]["label"]


# ---------------------------------------------------------------------------
# 1. detect + prepare
# ---------------------------------------------------------------------------

def stage_detect(ctx):
    tables: dict[str, pd.DataFrame] = ctx["tables"]
    best = None
    limit = pd.Timestamp.today().normalize() + pd.DateOffset(years=3)  # placeholder dates (e.g. year 2050) must not distort the span
    for name, df in tables.items():
        if len(df) < 30:
            continue
        tcands = detect_time_columns(df)
        if not tcands:
            continue
        for t in tcands[:8]:
            d = parse_time_column(df[t["name"]])
            d = d.where(d <= limit)
            ok = d.notna()
            if ok.mean() < 0.6:
                continue
            # finest frequency whose periods are (almost) all populated; a sparse or single-timestamp column never qualifies
            fkey, cov = None, 0.0
            for k in ("D", "W", "M", "Q", "Y"):
                pk = d[ok].dt.to_period(FREQ[k]["period"])
                span = (pk.max() - pk.min()).n + 1
                if span >= 12 and pk.nunique() / span >= 0.7:
                    fkey, cov = k, pk.nunique() / span
                    break
            if fkey is None:
                continue
            vals = detect_value_columns(df, exclude={t["name"]})
            if not vals:
                continue
            vc = max(vals, key=lambda v: (_rank_value(v["name"]), -v["missing_pct"]))["name"]
            nm = t["name"]
            score = (_rank_value(vc) * 10 + t["score"] + (3.0 if STRONG_PERIOD.search(nm) else 0.4 if WEAK_PERIOD.search(nm) else 0)
                     - (4.0 if BAD_TIME.search(nm) else 0) + cov + min(pk.nunique(), 60) / 12 + np.log10(len(df)))
            if best is None or score > best[0]:
                best = (score, name, nm, vc, fkey)
    if best is None:
        raise ValueError("No dataset has a date column with at least 12 distinct periods and a numeric measure to forecast.")
    _, table, tc, vc, key = best
    df = tables[table]
    d = parse_time_column(df[tc])
    v = df[vc]
    if not pd.api.types.is_numeric_dtype(v):
        v = pd.to_numeric(v.replace("", np.nan), errors="coerce")
    frame = pd.DataFrame({"d": d, "v": v.astype(float)}).dropna(subset=["d"])
    today = pd.Timestamp.today().normalize()
    limit = today + pd.DateOffset(years=3)
    dropped_future = int((frame.d > limit).sum())
    frame = frame[frame.d <= limit]

    cfg = FREQ[key]

    per = frame.d.dt.to_period(cfg["period"])
    g = frame.groupby(per).agg(value=("v", "sum"), rows=("v", "size"))
    full = pd.period_range(g.index.min(), g.index.max(), freq=cfg["period"])
    gaps = int(len(full) - len(g))
    g = g.reindex(full, fill_value=0)
    # trailing periods with far fewer rows than usual are incomplete — forecasting from them would bias the level down
    med_rows = float(g["rows"].replace(0, np.nan).median() or 1)
    incomplete = 0
    while len(g) > 12 and incomplete < 12 and g["rows"].iloc[-1] < 0.5 * med_rows:
        g = g.iloc[:-1]
        incomplete += 1
    if len(g) < 12:
        raise ValueError(f"Only {len(g)} {cfg['label']} periods of history — at least 12 are needed to forecast.")

    series = pd.DataFrame({"period": g.index.to_timestamp(), "value": g["value"].astype(float).to_numpy(), "rows": g["rows"].to_numpy()})
    y = series["value"].to_numpy()
    plot = series if len(series) <= 400 else series.iloc[:: int(np.ceil(len(series) / 400))]
    notes = []
    if incomplete:
        notes.append(P.note(f"{incomplete} trailing period(s) had under half the usual number of rows and were left out as incomplete.", "plain"))
    if gaps:
        notes.append(P.note(f"{gaps} period(s) had no rows and were treated as zero.", "plain"))
    if dropped_future:
        notes.append(P.note(f"{dropped_future:,} rows dated more than 3 years ahead were ignored as invalid dates.", "plain"))
    result = {
        "table": table, "time_col": tc, "value_col": vc, "frequency": key, "periods": int(len(series)),
        "start": series["period"].min(), "end": series["period"].max(), "total": float(y.sum()), "average": float(y.mean()),
        "gaps": gaps, "incomplete_dropped": incomplete,
        "page": P.page(
            P.kpis(("Table", table, f"{len(df):,} rows"), ("Measure", vc, f"summed by {cfg['label']} period"), ("Periods", str(len(series)), f"{series['period'].min():%b %Y} → {series['period'].max():%b %Y}"),
                   ("Average per period", P.fmt_num(float(y.mean())), f"total {P.fmt_num(float(y.sum()))}", True)),
            P.chart("line", f"{vc} by {cfg['label']} period", [{"period": f"{r.period:%Y-%m-%d}", "value": r.value} for r in plot.itertuples()], "period", [("value", vc, "brand")], fmt="num"),
            *notes),
    }
    working = pd.DataFrame({"Period": series["period"], vc: series["value"]})
    return clean(result), {"series": series, "freq": key, "m": cfg["m"], "working_frame": working}


# ---------------------------------------------------------------------------
# 2. explore
# ---------------------------------------------------------------------------

def stage_explore(ctx):
    from scipy import stats
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller

    s = ctx["art"]["series"]
    y = s["value"].to_numpy(float)
    n, m, key = len(y), ctx["art"]["m"], ctx["art"]["freq"]
    mean = float(np.mean(y)) or 1.0
    slope, _i, _r, p_lin, _se = stats.linregress(np.arange(n), y)
    trend_pct = float(slope / abs(mean))
    tau, p_mk = stats.kendalltau(np.arange(n), y)
    adf_p = float(adfuller(y)[1]) if n >= 10 else None

    seasonal_strength, phase, resid = None, None, None
    trend_c = pd.Series(y).rolling(max(3, min(m if m > 1 else 3, n // 3)), center=True, min_periods=1).mean().to_numpy()
    if m > 1 and n >= 2 * m:
        try:
            fit = STL(y, period=m, robust=True).fit()
            trend_c, seas, resid = fit.trend, fit.seasonal, fit.resid
            var_r, var_rs = float(np.var(resid)), float(np.var(resid + seas))
            seasonal_strength = float(max(0.0, 1 - var_r / var_rs)) if var_rs > 0 else 0.0
            ph = pd.DataFrame({"phase": np.arange(n) % m, "seasonal": seas}).groupby("phase")["seasonal"].mean()
            phase = [{"phase": int(k) + 1, "effect": float(v)} for k, v in ph.items()]
        except Exception:
            resid = None
    if resid is None:
        resid = y - trend_c
    sd = float(np.std(resid)) or 1.0
    z = (resid - np.mean(resid)) / sd
    anomalies = [{"period": f"{s['period'].iloc[i]:%Y-%m-%d}", "value": float(y[i]), "z": float(z[i])} for i in np.argsort(-np.abs(z)) if abs(z[i]) >= 3][:8]
    lags = min(24, n // 2 - 1)
    ac = acf(y, nlags=lags, fft=True).tolist() if lags >= 2 else []

    direction = "rising" if (p_mk < 0.05 and trend_pct > 0) else "falling" if (p_mk < 0.05 and trend_pct < 0) else "flat"
    blocks = [
        P.kpis(("Trend", f"{trend_pct * 100:+.1f}% / period", f"{direction} (Mann-Kendall p={p_mk:.3f})", direction == "falling"),
               ("Seasonality", f"{seasonal_strength * 100:.0f}%" if seasonal_strength is not None else "n/a", f"cycle of {m} periods" if m > 1 else "no seasonal cycle"),
               ("Stationary", "yes" if (adf_p is not None and adf_p < 0.05) else "no", f"ADF p={adf_p:.3f}" if adf_p is not None else ""),
               ("Unusual periods", str(len(anomalies)), "beyond 3 standard deviations")),
        P.chart("line", "Series and underlying trend", [{"period": f"{s['period'].iloc[i]:%Y-%m-%d}", "actual": float(y[i]), "trend": float(trend_c[i])} for i in range(n)], "period",
                [("actual", "Actual", "brand"), ("trend", "Trend", "pink")], height=260),
    ]
    if phase:
        blocks.append(P.chart("bar", "Seasonal pattern", phase, "phase", [("effect", "Effect vs trend", "brand")], subtitle="Average effect at each position in the cycle", height=200))
    if ac:
        blocks.append(P.chart("bar", "Autocorrelation", [{"lag": i, "acf": v} for i, v in enumerate(ac) if i > 0], "lag", [("acf", "Autocorrelation", "soft")], subtitle="How strongly each period relates to earlier ones", height=180))
    if anomalies:
        blocks.append(P.table("Unusual periods", ["Period", "Value", "Z-score"], [[a["period"], P.fmt_num(a["value"]), f"{a['z']:+.1f}"] for a in anomalies]))
    result = {"trend_pct": trend_pct, "trend_p": float(p_mk), "direction": direction, "seasonal_strength": seasonal_strength, "adf_p": adf_p,
              "anomalies": anomalies, "page": P.page(*blocks)}
    return clean(result), {}


# ---------------------------------------------------------------------------
# 3. models + rolling-origin backtest
# ---------------------------------------------------------------------------

def _fc_naive(y, h, m):
    return np.repeat(y[-1], h)


def _fc_snaive(y, h, m):
    if m <= 1 or len(y) < m:
        return None
    last = y[-m:]
    return np.array([last[i % m] for i in range(h)])


def _fc_drift(y, h, m):
    return y[-1] + (y[-1] - y[0]) / max(len(y) - 1, 1) * np.arange(1, h + 1)


def _fc_ets(y, h, m):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seas = "add" if (1 < m <= 12 and len(y) >= 2 * m) else None
    mod = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seas, seasonal_periods=m if seas else None, initialization_method="estimated")
    return np.asarray(mod.fit(optimized=True).forecast(h))


def _fc_arima(y, h, m):
    from statsmodels.tsa.arima.model import ARIMA

    seasonal = (0, 1, 1, m) if (1 < m <= 12 and len(y) >= 3 * m) else (0, 0, 0, 0)
    return np.asarray(ARIMA(y, order=(1, 1, 1), seasonal_order=seasonal).fit().forecast(h))


def _fc_hgb(y, h, m):
    from sklearn.ensemble import HistGradientBoostingRegressor

    L = int(min(12, max(3, len(y) // 4)))
    if len(y) <= L + 3:
        return None
    scale = float(np.mean(np.abs(y))) or 1.0
    z = y / scale

    def feats(hist, t):
        f = [hist[-k] for k in range(1, L + 1)]
        f.append(np.mean(hist[-L:]))
        if m > 1:
            f.append(t % m)
        return f

    X = np.array([feats(z[:t], t) for t in range(L, len(z))])
    T = z[L:]
    mod = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_depth=3, random_state=42).fit(X, T)
    hist, out = list(z), []
    for step in range(h):
        nxt = float(mod.predict(np.array([feats(np.array(hist), len(hist))]))[0])
        hist.append(nxt)
        out.append(nxt)
    return np.array(out) * scale


MODELS = {"Naive": _fc_naive, "Seasonal naive": _fc_snaive, "Drift": _fc_drift, "Exponential smoothing": _fc_ets, "ARIMA": _fc_arima, "Gradient boosting": _fc_hgb}


def _smape(a, f):
    den = np.abs(a) + np.abs(f)
    return float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(a - f) / np.where(den == 0, 1, den))))


def stage_models(ctx):
    s = ctx["art"]["series"]
    y = s["value"].to_numpy(float)
    n, m, key = len(y), ctx["art"]["m"], ctx["art"]["freq"]
    cfg = FREQ[key]
    h = cfg["h_bt"]
    min_train = max(12, 2 * m if m <= 12 else 12)
    h = int(max(1, min(h, (n - min_train) // 2))) if n - min_train < 2 * h else h
    K = int(min(3, max(1, (n - min_train) // h)))
    origins = [n - h * k for k in range(K, 0, -1)]
    if origins[0] < 8:
        raise ValueError("Not enough history to backtest.")

    board, errs, preds = [], {}, {}
    for name, fn in MODELS.items():
        t0 = time.time()
        try:
            fold_err, fold_pred, sm, ok = [], [], [], True
            for o in origins:
                f = fn(y[:o], h, m)
                if f is None or len(f) < h or not np.all(np.isfinite(f)):
                    ok = False
                    break
                a = y[o:o + h]
                f = np.asarray(f)[:len(a)]
                fold_err.append(a - f)
                fold_pred.append(f)
                sm.append(_smape(a, f))
            if not ok:
                continue
            E = np.array([np.pad(e, (0, h - len(e)), constant_values=np.nan) for e in fold_err])
            errs[name], preds[name] = E, fold_pred
            board.append({"model": name, "smape": float(np.mean(sm)), "mae": float(np.nanmean(np.abs(E))), "rmse": float(np.sqrt(np.nanmean(E ** 2))),
                          "fold_smape": [float(v) for v in sm], "seconds": round(time.time() - t0, 1), "note": MODEL_NOTES[name]})
        except Exception:
            continue
    if not board:
        raise RuntimeError("No model could be fitted to this series.")
    naive = next((b["smape"] for b in board if b["model"] == "Naive"), None)
    for b in board:
        b["vs_naive"] = (naive - b["smape"]) / naive if naive else None
    best = min(board, key=lambda b: b["smape"])
    for b in board:
        b["selected"] = b["model"] == best["model"]
    board.sort(key=lambda b: b["smape"])

    o = origins[-1]
    per = s["period"].to_numpy()
    bt = [{"period": f"{pd.Timestamp(per[i]):%Y-%m-%d}", "actual": float(y[i])} for i in range(max(0, o - 12), o)]
    for j in range(min(h, n - o)):
        bt.append({"period": f"{pd.Timestamp(per[o + j]):%Y-%m-%d}", "actual": float(y[o + j]), "forecast": float(preds[best["model"]][-1][j])})
    result = {
        "leaderboard": board, "best": best["model"], "horizon_backtest": h, "folds": K, "origins": [f"{pd.Timestamp(per[o]):%Y-%m-%d}" for o in origins],
        "naive_smape": naive,
        "page": P.page(
            P.kpis(("Best model", best["model"], MODEL_NOTES[best["model"]], True), ("Error (sMAPE)", P.pct(best["smape"]), "average over the backtests"),
                   ("Vs naive baseline", f"{best['vs_naive'] * 100:+.0f}%" if best["vs_naive"] is not None else "n/a", "positive = more accurate than repeating the last value"),
                   ("Backtests", f"{K} × {h} periods", "rolling origin, never sees the future")),
            P.table("Model comparison", ["Model", "sMAPE", "MAE", "RMSE", "Vs naive"],
                    [[("✓ " if b["selected"] else "") + b["model"], P.pct(b["smape"]), P.fmt_num(b["mae"]), P.fmt_num(b["rmse"]), f"{b['vs_naive'] * 100:+.0f}%" if b["vs_naive"] is not None else "—"] for b in board],
                    subtitle="Each model forecasts several periods it has not seen; lower error is better"),
            P.chart("bar", "Error by model", [{"model": b["model"], "smape": b["smape"]} for b in board], "model", [("smape", "sMAPE", "brand")], fmt="pct", height=200),
            P.chart("line", f"Backtest: {best['model']} vs actual", bt, "period", [("actual", "Actual", "brand"), ("forecast", "Forecast", "pink")], subtitle="Last backtest window", height=240)),
    }
    return clean(result), {"errors": errs[best["model"]], "h_bt": h, "best": best["model"]}


# ---------------------------------------------------------------------------
# 4. forecast
# ---------------------------------------------------------------------------

def stage_forecast(ctx):
    s = ctx["art"]["series"]
    y = s["value"].to_numpy(float)
    m, key, best = ctx["art"]["m"], ctx["art"]["freq"], ctx["art"]["best"]
    cfg = FREQ[key]
    H = cfg["h"]
    f = np.asarray(MODELS[best](y, H, m), dtype=float)
    E, h_bt = ctx["art"]["errors"], ctx["art"]["h_bt"]
    rmse_step = np.sqrt(np.nanmean(E ** 2, axis=0))
    floor = 0.02 * float(np.mean(np.abs(y)))
    sd = np.array([max(rmse_step[s_ - 1] if s_ <= h_bt else rmse_step[-1] * np.sqrt(s_ / h_bt), floor) for s_ in range(1, H + 1)])
    nonneg = bool(np.all(y >= 0))
    lo80, hi80, lo95, hi95 = f - 1.2816 * sd, f + 1.2816 * sd, f - 1.96 * sd, f + 1.96 * sd
    if nonneg:
        lo80, lo95 = np.maximum(lo80, 0), np.maximum(lo95, 0)
    last = pd.Period(s["period"].iloc[-1], freq=cfg["period"])
    future = pd.period_range(last + 1, periods=H, freq=cfg["period"]).to_timestamp()
    fc = pd.DataFrame({"period": future, "forecast": f, "lower_80": lo80, "upper_80": hi80, "lower_95": lo95, "upper_95": hi95})

    hist = s.tail(60)
    data = [{"period": f"{r.period:%Y-%m-%d}", "actual": float(r.value)} for r in hist.itertuples()]
    data[-1]["forecast"] = float(hist["value"].iloc[-1])
    data[-1]["band"] = [float(hist["value"].iloc[-1]), float(hist["value"].iloc[-1])]
    data += [{"period": f"{r.period:%Y-%m-%d}", "forecast": float(r.forecast), "band": [float(r.lower_80), float(r.upper_80)]} for r in fc.itertuples()]
    total_next = float(f.sum())
    prev = float(y[-H:].sum()) if len(y) >= H else None
    growth = (total_next / prev - 1) if prev else None
    result = {
        "model": best, "horizon": H, "next_period": float(f[0]), "next_total": total_next, "previous_total": prev, "growth": growth,
        "forecast": [{"period": f"{r.period:%Y-%m-%d}", "forecast": float(r.forecast), "lower_80": float(r.lower_80), "upper_80": float(r.upper_80)} for r in fc.itertuples()],
        "page": P.page(
            P.kpis(("Next period", P.fmt_num(float(f[0])), f"{future[0]:%b %Y}", True), (f"Next {H} periods", P.fmt_num(total_next), f"{future[0]:%b %Y} → {future[-1]:%b %Y}"),
                   (f"Vs previous {H}", f"{growth * 100:+.1f}%" if growth is not None else "n/a", f"{P.fmt_num(prev)} before" if prev else ""), ("Model", best, "chosen by backtest")),
            P.chart("line", "History and forecast", data, "period", [("actual", "Actual", "brand"), ("forecast", "Forecast", "pink")], subtitle="Shaded area = 80% interval", band=True, height=300),
            P.table("Forecast", ["Period", "Forecast", "80% range", "95% range"],
                    [[f"{r.period:%b %Y}", P.fmt_num(float(r.forecast)), f"{P.fmt_num(float(r.lower_80))} – {P.fmt_num(float(r.upper_80))}", f"{P.fmt_num(float(r.lower_95))} – {P.fmt_num(float(r.upper_95))}"] for r in fc.itertuples()][:24]),
            P.note("Intervals come from the model's errors on the backtests; they widen with distance. Treat the far end as a scenario, not a promise.", "plain")),
    }
    return clean(result), {"forecast_df": fc}


# ---------------------------------------------------------------------------
# 5. validate
# ---------------------------------------------------------------------------

def stage_validate(ctx):
    R = ctx["results"]
    mod, fc, det = R["models"], R["forecast"], R["detect"]
    best = next(b for b in mod["leaderboard"] if b["selected"])
    chk = []

    def add(name, status, detail):
        chk.append({"check": name, "status": status, "detail": detail})

    n = det["periods"]
    add("Enough history", "pass" if n >= 24 else "warn" if n >= 12 else "fail", f"{n} periods of history" + ("" if n >= 24 else " — 24 or more gives steadier results."))
    vn = best.get("vs_naive")
    add("Beats the naive baseline", "pass" if (vn is not None and vn >= 0.05) else "warn",
        f"{best['model']} is {vn * 100:+.0f}% more accurate than repeating the last value." if vn is not None else "Naive baseline unavailable.")
    fs = np.array(best["fold_smape"])
    cv = float(fs.std() / fs.mean()) if len(fs) > 1 and fs.mean() > 0 else 0.0
    add("Stable across backtests", "pass" if cv < 0.5 else "warn", f"Error per backtest: {', '.join(P.pct(v) for v in fs)}.")
    add("Error level", "pass" if best["smape"] < 0.15 else "warn" if best["smape"] < 0.30 else "fail", f"Average error {P.pct(best['smape'])} (sMAPE).")
    if det["incomplete_dropped"] or det["gaps"]:
        add("Data completeness", "warn", f"{det['incomplete_dropped']} incomplete trailing period(s) removed, {det['gaps']} empty period(s) filled with zero.")
    else:
        add("Data completeness", "pass", "No gaps or incomplete periods.")
    order = {"fail": 0, "warn": 1, "pass": 2}
    overall = min((c["status"] for c in chk), key=lambda s_: order[s_])
    result = {"checks": chk, "overall": overall, "passed": sum(c["status"] == "pass" for c in chk), "total": len(chk),
              "page": P.page(P.kpis(("Passed", f"{sum(c['status'] == 'pass' for c in chk)} of {len(chk)}"), ("Warnings", str(sum(c["status"] == "warn" for c in chk)), None, True), ("Failed", str(sum(c["status"] == "fail" for c in chk)))), P.checks(chk))}
    return clean(result), {}


# ---------------------------------------------------------------------------
# 6. deliverables + report
# ---------------------------------------------------------------------------

def stage_deliver(ctx):
    s, fc = ctx["art"]["series"], ctx["art"]["forecast_df"]
    vc = ctx["results"]["detect"]["value_col"]
    hist = pd.DataFrame({"period": s["period"], "kind": "history", "actual": s["value"]})
    out = pd.concat([hist, fc.assign(kind="forecast")], ignore_index=True)[["period", "kind", "actual", "forecast", "lower_80", "upper_80", "lower_95", "upper_95"]]
    out["period"] = out["period"].dt.strftime("%Y-%m-%d")
    mod = ctx["results"]["models"]
    bt = pd.DataFrame(mod["page"]["blocks"][-1]["data"]).rename(columns={"actual": "actual", "forecast": "backtest_forecast"})
    dictionary = pd.DataFrame([
        {"column": "period", "description": f"Start of the {_series_label(ctx['art']['freq'])} period."},
        {"column": "kind", "description": "history = observed; forecast = predicted."},
        {"column": "actual", "description": f"Observed {vc} summed over the period."},
        {"column": "forecast", "description": f"Predicted {vc} for the period ({mod['best']})."},
        {"column": "lower_80 / upper_80", "description": "80% prediction interval."},
        {"column": "lower_95 / upper_95", "description": "95% prediction interval."},
    ])
    result = {"rows": int(len(out)), "page": P.page(
        P.kpis(("Rows", f"{len(out):,}", "history + forecast"), ("Forecast rows", str(len(fc))), ("Backtest rows", str(len(bt)))),
        P.downloads([("enriched", "Forecast table (.csv)", "csv", True), ("enriched", "Forecast (.xlsx)", "xlsx"), ("accounts", "Backtest table (.csv)", "csv"), ("dictionary", "Column guide (.csv)", "csv"), ("report", "Report (.docx)", "docx")],
                    subtitle="History, forecast and intervals in one table"),
        P.table("Preview", ["Period", "Kind", "Actual", "Forecast", "80% low", "80% high"], [[r.period, r.kind, P.fmt_num(r.actual) if pd.notna(r.actual) else "", P.fmt_num(r.forecast) if pd.notna(r.forecast) else "", P.fmt_num(r.lower_80) if pd.notna(r.lower_80) else "", P.fmt_num(r.upper_80) if pd.notna(r.upper_80) else ""] for r in out.tail(12).itertuples()]))}
    return clean(result), {"enriched_csv": out.to_csv(index=False).encode(), "accounts_csv": bt.to_csv(index=False).encode(), "dictionary_csv": dictionary.to_csv(index=False).encode()}


def stage_report(ctx):
    R = ctx["results"]
    det, ex, mod, fc, val = R["detect"], R["explore"], R["models"], R["forecast"], R["validate"]
    best = next(b for b in mod["leaderboard"] if b["selected"])
    vc = det["value_col"]
    H = fc["horizon"]
    cycle = f"a {FREQ[det['frequency']]['m']}-period seasonal cycle" if (ex["seasonal_strength"] or 0) >= 0.3 else "no strong seasonality"
    bullets = [
        f"{vc} from {det['table']}, by {_series_label(det['frequency'])} period: {det['periods']} periods, average {P.fmt_num(det['average'])} per period.",
        f"The series is {ex['direction']} ({ex['trend_pct'] * 100:+.1f}% per period) with {cycle}.",
        f"{mod['best']} was the most accurate of {len(mod['leaderboard'])} models in rolling backtests (error {P.pct(best['smape'])}"
        + (f", {best['vs_naive'] * 100:+.0f}% vs the naive baseline" if best.get("vs_naive") is not None else "") + ").",
        f"Forecast for the next {H} periods: {P.fmt_num(fc['next_total'])}" + (f" ({fc['growth'] * 100:+.1f}% vs the previous {H})." if fc["growth"] is not None else "."),
    ]
    caveats = [c["detail"] for c in val["checks"] if c["status"] != "pass"]
    actions = []
    if ex["direction"] == "falling":
        actions.append({"driver": "Falling trend", "action": "Investigate the drivers of the decline before relying on the forecast to plan growth."})
    if (ex["seasonal_strength"] or 0) >= 0.3:
        actions.append({"driver": "Seasonality", "action": "Plan capacity, staffing and cash around the seasonal peaks and troughs shown in the pattern."})
    actions.append({"driver": "Uncertainty", "action": "Budget to the 80% range: plan costs to the upper end and revenue commitments to the lower end."})
    headline = {"key_result": f"Next {H} periods: {P.fmt_num(fc['next_total'])}" + (f" ({fc['growth'] * 100:+.1f}%)" if fc["growth"] is not None else ""),
                "model": mod["best"], "smape": best["smape"], "next_total": fc["next_total"], "growth": fc["growth"]}
    md = [f"## Executive summary\n\n" + "\n".join(f"- {b}" for b in bullets)]
    md.append("## Model comparison\n\n" + "| Model | Error (sMAPE) | Vs naive |\n|---|---|---|\n" + "\n".join(f"| {b['model']}{' ✓' if b['selected'] else ''} | {P.pct(b['smape'])} | {b['vs_naive'] * 100:+.0f}% |" if b.get("vs_naive") is not None else f"| {b['model']} | {P.pct(b['smape'])} | — |" for b in mod["leaderboard"]))
    md.append("## Forecast\n\n| Period | Forecast | 80% range |\n|---|---|---|\n" + "\n".join(f"| {r['period'][:7]} | {P.fmt_num(r['forecast'])} | {P.fmt_num(r['lower_80'])} – {P.fmt_num(r['upper_80'])} |" for r in fc["forecast"][:12]))
    md.append("## Validation\n\n| Check | Result | Detail |\n|---|---|---|\n" + "\n".join(f"| {c['check']} | {c['status'].upper()} | {c['detail']} |" for c in val["checks"]))
    md.append("## Notes\n\n- Intervals come from backtest errors and widen with distance.\n- A forecast extends patterns in the data; it cannot anticipate events the data has not seen.")
    markdown = "\n\n".join(md) + "\n"
    narrative = {"executive_summary": " ".join(bullets), "actions": actions, "caveats": caveats, "source": "template"}
    result = {"headline": headline, "narrative": narrative, "markdown": markdown,
              "page": P.page(P.kpis(("Next period", P.fmt_num(fc["next_period"]), None, True), (f"Next {H} periods", P.fmt_num(fc["next_total"]), f"{fc['growth'] * 100:+.1f}% vs previous" if fc["growth"] is not None else None),
                                    ("Model", mod["best"]), ("Error (sMAPE)", P.pct(best["smape"]))),
                             P.bullets("Summary", bullets), *[P.note(c) for c in caveats],
                             P.bullets("Recommended actions", [f"{a['driver']} — {a['action']}" for a in actions]))}
    return clean(result), {}


STAGES = [
    ("detect", "Prepare the series", stage_detect, True),
    ("explore", "Explore the series", stage_explore, False),
    ("models", "Backtest models", stage_models, False),
    ("forecast", "Forecast", stage_forecast, False),
    ("validate", "Validate", stage_validate, False),
    ("deliver", "Build deliverables", stage_deliver, False),
    ("report", "Write summary", stage_report, False),
]
