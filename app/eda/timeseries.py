"""
Enterprise-grade Time Series Analysis Module
AutoEDA Platform — Backend

Covers all standard EDA phases for time series DS/ML projects:
  - Data quality & completeness
  - Stationarity & unit root tests (ADF, KPSS, PP)
  - Trend & seasonality detection + decomposition
  - Autocorrelation diagnostics (ACF/PACF)
  - Change point detection
  - Anomaly/outlier detection (rolling Z-score + IQR fence)
  - Granger causality (multi-column)
  - Lag feature & correlation analysis
  - Distribution diagnostics (normality, skew, kurtosis)
  - Forecasting readiness summary
  - Spectral analysis (FFT dominant frequencies)
  - Rolling statistics & volatility
  - All results are JSON-serialisable, truncated for low-latency delivery
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_POINTS = 2_000  # max points sent to frontend for rendering


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except Exception:
        return None


def _safe_list(arr, n: int = MAX_POINTS) -> list[float | None]:
    return [_safe(v) for v in np.asarray(arr, dtype=float)[:n]]


def _downsample(dates: list, values: list, n: int = MAX_POINTS):
    """LTTB-inspired uniform downsample preserving shape."""
    if len(dates) <= n or len(dates) != len(values):
        return dates, values
    idx = np.round(np.linspace(0, len(dates) - 1, n)).astype(int)
    return [dates[i] for i in idx], [values[i] for i in idx]


def _fmt_dates(series: pd.Series) -> list[str]:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()


def _adaptive_max_lag(n: int, base_fraction: int = 25) -> int:
    """Compute adaptive max lag for Granger causality based on data size."""
    return min(12, max(1, n // base_fraction))


def _adaptive_ewma_alpha(period: int = 10) -> float:
    """Standard EWMA alpha using smoothing factor formula: 2/(n+1)."""
    return 2.0 / (period + 1)


# ---------------------------------------------------------------------------
# Sub-analysis helpers
# ---------------------------------------------------------------------------


def _data_quality(df_ts: pd.DataFrame, time_col: str, value_col: str) -> dict:
    n_total = len(df_ts)
    n_missing = int(df_ts[value_col].isna().sum())
    n_duplicates = int(df_ts[time_col].duplicated().sum())
    series = df_ts[value_col].dropna().astype(float)

    # Temporal regularity
    deltas = df_ts[time_col].diff().dropna()
    delta_seconds = deltas.dt.total_seconds()
    freq_mode = delta_seconds.mode().iloc[0] if len(delta_seconds) > 0 else None
    irregular_pct = float((delta_seconds != freq_mode).mean() * 100) if freq_mode and freq_mode > 0 else None

    # Inferred frequency label
    freq_label = None
    if freq_mode is not None and freq_mode > 0:
        s = freq_mode
        if s < 60:
            freq_label = f"{int(s)}s"
        elif s < 3600:
            freq_label = f"{int(s/60)}min"
        elif s < 86400:
            freq_label = f"{int(s/3600)}h"
        elif s < 86400 * 7:
            freq_label = f"{int(s/86400)}D"
        elif s < 86400 * 32:
            freq_label = f"{int(s/86400/7)}W"
        else:
            freq_label = f"{int(s/86400/30)}M"

    # Gaps (consecutive missing periods > 2× modal delta)
    gap_list = []
    if freq_mode and freq_mode > 0:
        gaps = deltas[delta_seconds > 2 * freq_mode]
        gap_list = [
            {
                "start": df_ts.loc[i - 1, time_col].strftime("%Y-%m-%dT%H:%M:%S"),
                "end": df_ts.loc[i, time_col].strftime("%Y-%m-%dT%H:%M:%S"),
                "gap_seconds": _safe(delta_seconds.loc[i]),
            }
            for i in gaps.index[:10]
        ]

    # Constant segments
    n_zeros = int((series == 0).sum())
    n_const_runs = 0
    if len(series) > 1:
        diff = series.diff().ne(0)
        runs = diff.cumsum()
        run_lengths = runs.value_counts()
        n_const_runs = int((run_lengths >= 5).sum())

    return {
        "n_total": n_total,
        "n_missing": n_missing,
        "missing_pct": round(n_missing / max(n_total, 1) * 100, 2),
        "n_duplicates": n_duplicates,
        "n_zeros": n_zeros,
        "n_const_runs": n_const_runs,
        "irregular_pct": _safe(irregular_pct),
        "inferred_freq": freq_label,
        "freq_seconds": _safe(freq_mode),
        "gaps": gap_list,
    }


def _descriptive_stats(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return {}
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    return {
        "count": int(len(s)),
        "mean": _safe(s.mean()),
        "median": _safe(s.median()),
        "std": _safe(s.std()),
        "min": _safe(s.min()),
        "max": _safe(s.max()),
        "range": _safe(float(s.max()) - float(s.min())),
        "q1": _safe(q1),
        "q3": _safe(q3),
        "iqr": _safe(iqr),
        "skewness": _safe(s.skew()),
        "kurtosis": _safe(s.kurt()),
        "cv": _safe(s.std() / s.mean() * 100 if s.mean() != 0 else None),
    }


def _normality_tests(series: pd.Series) -> dict:
    s = series.dropna().astype(float).values
    result: dict[str, Any] = {}
    if len(s) < 8:
        return result
    try:
        from scipy.stats import jarque_bera, shapiro, normaltest

        if len(s) <= 5000:
            stat, p = shapiro(s[:5000])
            result["shapiro"] = {"statistic": _safe(stat), "pvalue": _safe(p), "is_normal": bool(p > 0.05)}

        stat, p = jarque_bera(s)
        result["jarque_bera"] = {"statistic": _safe(stat), "pvalue": _safe(p), "is_normal": bool(p > 0.05)}

        stat, p = normaltest(s)
        result["dagostino"] = {"statistic": _safe(stat), "pvalue": _safe(p), "is_normal": bool(p > 0.05)}
    except ImportError:
        pass
    return result


def _stationarity_tests(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    result: dict[str, Any] = {"adf": None, "kpss": None, "zivot_andrews": None, "verdict": "unknown"}

    # ADF
    try:
        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(s, autolag="AIC")
        adf_stat, adf_p = float(adf[0]), float(adf[1])
        result["adf"] = {
            "statistic": _safe(adf_stat),
            "pvalue": _safe(adf_p),
            "critical_values": {k: _safe(v) for k, v in adf[4].items()},
            "is_stationary": bool(adf_p < 0.05),
            "interpretation": "Stationary (reject H₀ of unit root)" if adf_p < 0.05 else "Non-stationary (fail to reject H₀)",
        }
    except Exception:
        pass

    # KPSS
    try:
        from statsmodels.tsa.stattools import kpss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_res = kpss(s, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
        result["kpss"] = {
            "statistic": _safe(kpss_stat),
            "pvalue": _safe(kpss_p),
            "critical_values": {k: _safe(v) for k, v in kpss_res[3].items()},
            "is_stationary": bool(kpss_p > 0.05),
            "interpretation": "Stationary (fail to reject H₀ of stationarity)" if kpss_p > 0.05 else "Non-stationary (reject H₀)",
        }
    except Exception:
        pass

    # Zivot-Andrews structural break test
    try:
        from statsmodels.tsa.stattools import zivot_andrews

        za = zivot_andrews(s)
        result["zivot_andrews"] = {
            "statistic": _safe(float(za[0])),
            "pvalue": _safe(float(za[1])),
            "break_index": int(za[4]) if len(za) > 4 else None,
            "interpretation": "Break detected — consider structural break modelling" if float(za[1]) < 0.05 else "No significant structural break",
        }
    except Exception:
        pass

    # Verdict logic
    adf_stat = result["adf"]["is_stationary"] if result["adf"] else None
    kpss_stat = result["kpss"]["is_stationary"] if result["kpss"] else None
    if adf_stat is not None and kpss_stat is not None:
        if adf_stat and kpss_stat:
            verdict = "stationary"
        elif not adf_stat and not kpss_stat:
            verdict = "non_stationary"
        else:
            verdict = "mixed"
    elif adf_stat is not None:
        verdict = "stationary" if adf_stat else "non_stationary"
    elif kpss_stat is not None:
        verdict = "stationary" if kpss_stat else "non_stationary"
    else:
        verdict = "unknown"
    result["verdict"] = verdict

    return result


def _differencing_suggestions(series: pd.Series) -> dict:
    """Suggest differencing orders needed to achieve stationarity."""
    s = series.dropna().astype(float)
    suggestions: dict[str, Any] = {}
    if len(s) < 10:
        return suggestions
    try:
        from statsmodels.tsa.stattools import ndiffs, acf

        d = ndiffs(s, test="adf")
        suggestions["regular_diff_order"] = int(d)
        
        # Seasonal diff order (heuristic using ACF peak)
        acf_vals = acf(s, nlags=min(50, len(s) // 3))
        # Find dominant seasonal lag (peak beyond lag 1)
        if len(acf_vals) > 2:
            peak_lag = int(np.argmax(np.abs(acf_vals[2:])) + 2)
            suggestions["suggested_seasonal_period"] = peak_lag
    except Exception:
        pass
    return suggestions


def _decomposition(series: pd.Series, dates: list[str], detected_period: int | None) -> dict | None:
    if len(series) < 24 or len(dates) != len(series):
        return None
    try:
        from statsmodels.tsa.seasonal import STL, seasonal_decompose

        # Adaptive period calculation
        if detected_period is not None and detected_period > 1:
            period = detected_period
        else:
            period = max(2, min(int(len(series) / 10), 365))
        
        # Ensure period doesn't exceed series length
        period = min(period, len(series) // 2)
        
        # Prefer STL (robust)
        try:
            stl = STL(series, period=period, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid
            method = "STL"
        except Exception:
            res = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid
            method = "additive"

        # Variance explained
        trend_strength = seasonal_strength = None
        try:
            total_var = float(series.var())
            if total_var > 0:
                trend_var = float(pd.Series(trend).dropna().var())
                seasonal_var = float(pd.Series(seasonal).dropna().var())
                trend_strength = round(min(100, max(0, (trend_var / total_var) * 100)), 1)
                seasonal_strength = round(min(100, max(0, (seasonal_var / total_var) * 100)), 1)
        except Exception:
            pass

        dates_ds, trend_ds = _downsample(dates, trend.tolist())
        _, seasonal_ds = _downsample(dates, seasonal.tolist())
        _, resid_ds = _downsample(dates, residual.tolist())

        return {
            "method": method,
            "period": period,
            "trend": _safe_list(trend_ds),
            "seasonal": _safe_list(seasonal_ds),
            "residual": _safe_list(resid_ds),
            "dates": dates_ds,
            "trend_strength_pct": _safe(trend_strength),
            "seasonal_strength_pct": _safe(seasonal_strength),
        }
    except Exception:
        return None


def _acf_pacf(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    result: dict[str, Any] = {}
    if len(s) < 10:
        return result
    try:
        from statsmodels.tsa.stattools import acf, pacf

        n_lags = min(60, max(10, len(s) // 3))
        conf_int_acf = 1.96 / math.sqrt(len(s))
        acf_vals, acf_ci = acf(s, nlags=n_lags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(s, nlags=n_lags, alpha=0.05)

        # Significant lags (outside 95% CI)
        sig_acf = [int(i) for i, v in enumerate(acf_vals) if abs(v) > conf_int_acf and i > 0]
        sig_pacf = [int(i) for i, v in enumerate(pacf_vals) if abs(v) > conf_int_acf and i > 0]

        result = {
            "acf": {
                "values": _safe_list(acf_vals),
                "ci_upper": [_safe(ci[1]) for ci in acf_ci] if acf_ci is not None else [],
                "ci_lower": [_safe(ci[0]) for ci in acf_ci] if acf_ci is not None else [],
                "significant_lags": sig_acf[:20],
                "conf_threshold": _safe(conf_int_acf),
            },
            "pacf": {
                "values": _safe_list(pacf_vals),
                "ci_upper": [_safe(ci[1]) for ci in pacf_ci] if pacf_ci is not None else [],
                "ci_lower": [_safe(ci[0]) for ci in pacf_ci] if pacf_ci is not None else [],
                "significant_lags": sig_pacf[:20],
            },
            "n_lags": n_lags,
        }

        # ARIMA order suggestions
        p_order = sig_pacf[0] if sig_pacf else 0
        q_order = sig_acf[0] if sig_acf else 0
        result["arima_hints"] = {
            "suggested_p": p_order,
            "suggested_q": q_order,
            "note": f"PACF cuts off at lag {p_order} → AR({p_order}); ACF cuts off at lag {q_order} → MA({q_order})",
        }
    except Exception:
        pass
    return result


def _spectral_analysis(series: pd.Series, freq_seconds: float | None) -> dict:
    """FFT to detect dominant periodicities."""
    s = series.dropna().astype(float).values
    if len(s) < 20:
        return {}
    try:
        fft_vals = np.abs(np.fft.rfft(s - s.mean()))
        freqs = np.fft.rfftfreq(len(s))
        
        # Top 5 dominant frequencies (skip DC component)
        if len(fft_vals) > 1:
            idx = np.argsort(fft_vals[1:])[::-1][:5] + 1
        else:
            idx = []
            
        dominant = []
        for i in idx:
            if i < len(freqs):
                freq = float(freqs[i])
                period_pts = round(1.0 / freq) if freq > 0 else None
                period_label = None
                if period_pts and freq_seconds and freq_seconds > 0:
                    secs = period_pts * freq_seconds
                    if secs < 3600:
                        period_label = f"{int(secs/60)}min"
                    elif secs < 86400:
                        period_label = f"{round(secs/3600,1)}h"
                    elif secs < 86400 * 8:
                        period_label = f"{round(secs/86400,1)}D"
                    elif secs < 86400 * 370:
                        period_label = f"{round(secs/86400/7,1)}W"
                    else:
                        period_label = f"{round(secs/86400/365,1)}Y"
                dominant.append({
                    "frequency": _safe(freq),
                    "amplitude": _safe(float(fft_vals[i])),
                    "period_points": period_pts,
                    "period_label": period_label,
                })
        return {"dominant_frequencies": dominant}
    except Exception:
        return {}


def _change_point_detection(series: pd.Series, dates: list[str]) -> dict:
    """PELT change point detection via ruptures (optional dep) or CUSUM fallback."""
    s = series.dropna().astype(float).values
    result: dict[str, Any] = {"method": None, "change_points": []}
    if len(s) < 30 or len(dates) != len(series.dropna()):
        return result

    # Try ruptures (fast C backend)
    try:
        import ruptures as rpt

        model = rpt.Pelt(model="rbf", min_size=max(2, len(s) // 20)).fit(s)
        pen = 3 * np.std(s)
        bkps = model.predict(pen=pen)
        cps = [int(b - 1) for b in bkps[:-1] if 0 <= b - 1 < len(dates)]
        result["method"] = "PELT (rbf)"
        result["change_points"] = [
            {"index": i, "date": dates[i], "value": _safe(float(s[i]))} for i in cps[:20]
        ]
        return result
    except ImportError:
        pass

    # CUSUM fallback
    try:
        mean = s.mean()
        std_s = np.std(s)
        if std_s == 0:
            return result
        cusum_pos = np.zeros(len(s))
        cusum_neg = np.zeros(len(s))
        k = 0.5 * std_s
        for i in range(1, len(s)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (s[i] - mean) - k)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - (s[i] - mean) - k)
        threshold = 5 * std_s
        cp_mask = (cusum_pos > threshold) | (cusum_neg > threshold)
        # Find first crossing per run
        change_pts = []
        in_run = False
        for i, flag in enumerate(cp_mask):
            if flag and not in_run:
                change_pts.append(i)
                in_run = True
            elif not flag:
                in_run = False
        result["method"] = "CUSUM"
        result["change_points"] = [
            {"index": i, "date": dates[i], "value": _safe(float(s[i]))} for i in change_pts[:20] if i < len(dates)
        ]
    except Exception:
        pass
    return result


def _anomaly_detection(series: pd.Series, dates: list[str], window: int) -> dict:
    """Multi-method anomaly detection: rolling Z-score + IQR fence."""
    s = series.astype(float)
    result: dict[str, Any] = {"rolling_zscore": [], "iqr_fence": [], "combined": []}

    if len(s) < 10 or len(dates) != len(s):
        result["total_anomalies"] = 0
        return result

    # Rolling Z-score
    rm = s.rolling(window=window, center=True, min_periods=1).mean()
    rs = s.rolling(window=window, center=True, min_periods=1).std().fillna(1)
    z_scores = (s - rm) / (rs + 1e-9)
    zscore_mask = z_scores.abs() > 3
    result["rolling_zscore"] = [
        {"index": int(i), "date": dates[i], "value": _safe(float(s[i])), "z_score": _safe(float(z_scores[i]))}
        for i in series.index[zscore_mask]
    ][:50]

    # IQR fence
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    iqr_mask = (s < lower) | (s > upper)
    result["iqr_fence"] = [
        {"index": int(i), "date": dates[i], "value": _safe(float(s[i])), "fence_lower": _safe(float(lower)), "fence_upper": _safe(float(upper))}
        for i in series.index[iqr_mask]
    ][:50]

    # Union
    all_idx = set(x["index"] for x in result["rolling_zscore"]) | set(x["index"] for x in result["iqr_fence"])
    result["combined"] = sorted(all_idx)
    result["total_anomalies"] = len(all_idx)

    return result


def _rolling_stats(series: pd.Series, dates: list[str]) -> dict:
    """Multi-window rolling statistics for volatility analysis."""
    s = series.astype(float)
    if len(s) < 5 or len(dates) != len(s):
        return {"windows": {}, "ewma": {}, "volatility": {}}
    
    # Adaptive window sizes based on series length
    windows = [
        max(3, len(series) // 20),
        max(7, len(series) // 10),
        max(14, len(series) // 5),
    ]
    windows = sorted(set(int(w) for w in windows if w > 0))

    result: dict[str, Any] = {"windows": {}}
    for w in windows:
        rm = s.rolling(window=w, min_periods=1).mean()
        rs = s.rolling(window=w, min_periods=1).std()
        dates_ds, mean_ds = _downsample(dates, rm.tolist())
        _, std_ds = _downsample(dates, rs.tolist())
        result["windows"][str(w)] = {
            "mean": _safe_list(mean_ds),
            "std": _safe_list(std_ds),
            "dates": dates_ds,
        }

    # EWMA with adaptive alpha
    alpha = _adaptive_ewma_alpha(period=max(2, len(s) // 10))
    ewma = s.ewm(alpha=alpha, adjust=False).mean()
    dates_ds, ewma_ds = _downsample(dates, ewma.tolist())
    result["ewma"] = {"alpha": round(alpha, 4), "values": _safe_list(ewma_ds), "dates": dates_ds}

    # Rolling volatility (annualized std)
    vol_window = max(5, len(series) // 15)
    log_returns = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan)
    rolling_vol = log_returns.rolling(vol_window).std() * math.sqrt(252)
    _, vol_ds = _downsample(dates, rolling_vol.tolist())
    result["volatility"] = {"window": vol_window, "values": _safe_list(vol_ds), "dates": dates_ds}

    return result


def _lag_analysis(series: pd.Series) -> dict:
    """Lag feature correlation matrix (up to 20 lags)."""
    s = series.dropna().astype(float)
    n_lags = min(20, max(1, len(s) // 5))
    result: dict[str, Any] = {"lags": [], "pearson": [], "spearman": []}
    if len(s) < 10:
        return result
    try:
        from scipy.stats import spearmanr

        for lag in range(1, n_lags + 1):
            shifted = s.shift(lag)
            mask = ~(s.isna() | shifted.isna())
            if mask.sum() < 10:
                break
            pearson_r = float(s[mask].corr(shifted[mask]))
            sp_r, _ = spearmanr(s[mask], shifted[mask])
            result["lags"].append(lag)
            result["pearson"].append(_safe(pearson_r))
            result["spearman"].append(_safe(float(sp_r)))
    except Exception:
        pass
    return result


def _trend_tests(series: pd.Series) -> dict:
    """Mann-Kendall trend test + Sen's slope."""
    s = series.dropna().astype(float)
    result: dict[str, Any] = {}
    if len(s) < 10:
        return result
    try:
        import pymannkendall as mk

        res = mk.original_test(s)
        result["mann_kendall"] = {
            "trend": res.trend,
            "p_value": _safe(res.p),
            "tau": _safe(res.Tau),
            "sen_slope": _safe(res.slope),
            "interpretation": f"{res.trend.capitalize()} trend (p={round(res.p,4)})",
        }
        return result
    except ImportError:
        pass

    # Fallback: linear regression slope significance
    try:
        from scipy.stats import kendalltau, linregress

        x = np.arange(len(s))
        slope, intercept, r, p, se = linregress(x, s.values)
        result["linear_trend"] = {
            "slope": _safe(float(slope)),
            "pvalue": _safe(float(p)),
            "r_squared": _safe(float(r**2)),
            "interpretation": ("Significant trend" if float(p) < 0.05 else "No significant trend")
            + f" (slope={round(float(slope),4)}, p={round(float(p),4)})",
        }
        tau, p_kt = kendalltau(x, s.values)
        result["kendall_tau"] = {"tau": _safe(float(tau)), "pvalue": _safe(float(p_kt))}
    except Exception:
        pass
    return result


def _seasonality_tests(series: pd.Series, period: int | None = None) -> dict:
    """ACF-based seasonality detection and seasonal differencing order."""
    s = series.dropna().astype(float)
    result: dict[str, Any] = {}
    if len(s) < 12:
        return result
    
    try:
        from statsmodels.tsa.stattools import acf

        max_lag = min(365, max(12, len(s) // 3))
        acf_vals = acf(s, nlags=max_lag)

        # Find peaks in ACF for seasonality detection
        if period is None:
            peaks = []
            for lag in range(2, len(acf_vals)):
                if (
                    abs(acf_vals[lag]) > 0.2
                    and lag > 1
                    and abs(acf_vals[lag]) > abs(acf_vals[lag - 1])
                    and (lag + 1 >= len(acf_vals) or abs(acf_vals[lag]) > abs(acf_vals[lag + 1]))
                ):
                    peaks.append((lag, abs(acf_vals[lag])))

            if peaks:
                period = sorted(peaks, key=lambda x: x[1], reverse=True)[0][0]

        # Seasonal strength from ACF at period lag
        if period and period < len(acf_vals):
            result["acf_seasonal_strength"] = _safe(float(acf_vals[period]))
            result["has_seasonality"] = bool(abs(float(acf_vals[period])) > 0.2)
            result["dominant_period"] = period
        else:
            result["has_seasonality"] = False

        # Seasonal differencing order
        try:
            from statsmodels.tsa.stattools import nsdiffs

            D = nsdiffs(s, m=period if period else 12, test="ocsb")
            result["seasonal_diff_order"] = int(D)
        except Exception:
            pass
    except Exception:
        pass

    return result


def _granger_causality(df: pd.DataFrame, target_col: str, other_cols: list[str]) -> dict:
    """Granger causality test of other numeric columns on the target."""
    result: dict[str, Any] = {}
    if not other_cols or len(df) < 30:
        return result
    
    max_lag = _adaptive_max_lag(len(df))
    
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        for col in other_cols[:5]:
            try:
                combined = df[[target_col, col]].dropna()
                if len(combined) < max_lag * 5:
                    continue
                test_res = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
                # Extract min p-value across lags
                min_p = min(float(test_res[lag][0]["ssr_chi2test"][1]) for lag in test_res)
                best_lag = min(test_res, key=lambda lag: float(test_res[lag][0]["ssr_chi2test"][1]))
                result[col] = {
                    "min_pvalue": _safe(min_p),
                    "best_lag": int(best_lag),
                    "granger_causes_target": bool(min_p < 0.05),
                }
            except Exception:
                continue
    except ImportError:
        pass
    return result


def _forecasting_readiness(
    stationarity: dict,
    decomp: dict | None,
    acf_pacf: dict,
    data_quality: dict,
    trend_tests: dict,
    seasonality: dict,
) -> dict:
    """Generate a model recommendation summary for forecasting."""
    issues = []
    recommendations = []

    missing_pct = data_quality.get("missing_pct", 0)
    if missing_pct and missing_pct > 5:
        issues.append(f"High missing data ({missing_pct:.1f}%) — impute before modelling")
    if data_quality.get("irregular_pct") and data_quality["irregular_pct"] > 10:
        issues.append("Irregular timestamps — resample to regular frequency")
    if data_quality.get("n_duplicates", 0) > 0:
        issues.append(f"{data_quality['n_duplicates']} duplicate timestamps — deduplicate first")

    verdict = stationarity.get("verdict", "unknown")
    d_order = 0
    if verdict in ("non_stationary", "mixed"):
        issues.append("Series is non-stationary — differencing required")
        d_order = 1

    has_seasonality = seasonality.get("has_seasonality", False)
    D_order = seasonality.get("seasonal_diff_order", 0)

    trend_str = None
    if "mann_kendall" in trend_tests:
        trend_str = trend_tests["mann_kendall"].get("trend")
    elif "linear_trend" in trend_tests:
        slope = trend_tests["linear_trend"].get("slope", 0)
        trend_str = "increasing" if (slope or 0) > 0 else "decreasing"

    p_hint = acf_pacf.get("arima_hints", {}).get("suggested_p", 0)
    q_hint = acf_pacf.get("arima_hints", {}).get("suggested_q", 0)

    if has_seasonality:
        seasonal_period = seasonality.get("dominant_period", 12)
        recommendations.append({
            "model": "SARIMA",
            "params": f"SARIMA(p={p_hint},d={d_order},q={q_hint})×(P,D={D_order},Q,m={seasonal_period})",
            "rationale": "Seasonal patterns detected; SARIMA captures both seasonal and non-seasonal dynamics",
            "priority": 1,
        })
        recommendations.append({
            "model": "Prophet",
            "params": "Auto (seasonal_mode=additive)",
            "rationale": "Handles multiple seasonalities, holidays, trend changepoints automatically",
            "priority": 2,
        })
    else:
        recommendations.append({
            "model": "ARIMA",
            "params": f"ARIMA({p_hint},{d_order},{q_hint})",
            "rationale": f"No clear seasonality; ARIMA captures {'trend + ' if trend_str else ''}autocorrelation structure",
            "priority": 1,
        })

    if decomp:
        ts = decomp.get("trend_strength_pct", 0) or 0
        ss = decomp.get("seasonal_strength_pct", 0) or 0
        if ts > 50 or ss > 50:
            recommendations.append({
                "model": "Exponential Smoothing (ETS)",
                "params": "ETS(Error,Trend,Season) — auto selection",
                "rationale": f"Strong trend ({ts:.0f}%) or seasonal ({ss:.0f}%) components favour ETS",
                "priority": 3,
            })

    recommendations.append({
        "model": "XGBoost / LightGBM with lag features",
        "params": f"lag_features=[1..{min(p_hint+4, 20)}], rolling_mean, rolling_std",
        "rationale": "ML models excel on large datasets with exogenous features; no stationarity requirement",
        "priority": 4,
    })

    return {
        "issues": issues,
        "recommendations": sorted(recommendations, key=lambda x: x["priority"]),
        "suggested_arima_order": (p_hint, d_order, q_hint),
        "stationarity_verdict": verdict,
        "has_trend": bool(trend_str and trend_str not in ("no trend", None)),
        "has_seasonality": has_seasonality,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> dict:
    """
    Full enterprise-grade time series analysis.
    Returns a JSON-serialisable dict compatible with the existing API contract.
    All new keys are additive — backwards compatible.
    """
    ERROR_TEMPLATE = {
        "time_col": time_col, "value_col": value_col, "n_points": 0,
        "start_date": "", "end_date": "", "has_trend": False,
        "line_data": {}, "rolling": {}, "anomalies": [],
    }

    # Column validation
    if time_col not in df.columns or value_col not in df.columns:
        return {**ERROR_TEMPLATE, "error": "Invalid columns"}

    df_ts = df[[time_col, value_col]].copy()
    try:
        df_ts[time_col] = pd.to_datetime(df_ts[time_col])
    except Exception:
        return {**ERROR_TEMPLATE, "error": f"Cannot parse '{time_col}' as datetime"}

    df_ts = df_ts.sort_values(time_col).reset_index(drop=True)
    series_raw = df_ts[value_col]

    if series_raw.dropna().__len__() < 10:
        return {**ERROR_TEMPLATE, "n_points": len(df_ts), "error": "Need at least 10 non-null data points"}

    series = (
        series_raw
        .astype(float)
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    dates = _fmt_dates(df_ts[time_col])

    # Downsampled line data
    dates_ds, values_ds = _downsample(dates, series.tolist())
    line_data = {"dates": dates_ds, "values": _safe_list(values_ds)}

    # Adaptive rolling window
    window = max(3, len(series) // 20)

    # --- Run all analyses ---
    quality = _data_quality(df_ts, time_col, value_col)
    freq_seconds = quality.get("freq_seconds")
    stats = _descriptive_stats(series)
    normality = _normality_tests(series)
    stationarity = _stationarity_tests(series)
    diff_suggestions = _differencing_suggestions(series)
    seasonality = _seasonality_tests(series)
    decomp = _decomposition(
        series,
        dates,
        seasonality.get("dominant_period")
    )
    acf_pacf = _acf_pacf(series)
    spectral = _spectral_analysis(series, freq_seconds)
    change_points = _change_point_detection(series, dates)
    anomalies_full = _anomaly_detection(series, dates, window)
    rolling = _rolling_stats(series, dates)
    lag_analysis = _lag_analysis(series)
    trend_tests = _trend_tests(series)

    # Granger causality (other numeric columns vs target)
    other_numeric = [
        c for c in df.columns
        if c != value_col and c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    granger = _granger_causality(df, value_col, other_numeric)

    readiness = _forecasting_readiness(stationarity, decomp, acf_pacf, quality, trend_tests, seasonality)

    # --- Assemble response (backwards-compatible) ---
    return {
        # Legacy fields (unchanged contract)
        "time_col": time_col,
        "value_col": value_col,
        "n_points": len(series),
        "start_date": dates[0],
        "end_date": dates[-1],
        "has_trend": readiness["has_trend"],
        "seasonality": seasonality.get("dominant_period"),
        "adf_statistic": (stationarity.get("adf") or {}).get("statistic"),
        "adf_pvalue": (stationarity.get("adf") or {}).get("pvalue"),
        "is_stationary": (stationarity.get("adf") or {}).get("is_stationary"),
        "line_data": line_data,
        "rolling": {  # Legacy single-window rolling (kept for backwards compat)
            "window": window,
            "mean": rolling["windows"].get(str(window), {}).get("mean", []),
            "std": rolling["windows"].get(str(window), {}).get("std", []),
        },
        "decomposition": decomp,
        "acf": acf_pacf.get("acf"),
        "pacf": acf_pacf.get("pacf"),
        "anomalies": anomalies_full.get("rolling_zscore", []),  # legacy field

        # --- New enterprise fields ---
        "data_quality": quality,
        "descriptive_stats": stats,
        "normality_tests": normality,
        "stationarity": stationarity,
        "differencing_suggestions": diff_suggestions,
        "acf_pacf_full": acf_pacf,
        "spectral": spectral,
        "change_points": change_points,
        "anomalies_full": anomalies_full,
        "rolling_full": rolling,
        "lag_analysis": lag_analysis,
        "trend_tests": trend_tests,
        "seasonality_tests": seasonality,
        "granger_causality": granger,
        "forecasting_readiness": readiness,
    }