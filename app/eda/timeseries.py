import numpy as np
import pandas as pd


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def run_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> dict:
    if time_col not in df.columns or value_col not in df.columns:
        return {
            "time_col": time_col, "value_col": value_col, "n_points": 0,
            "start_date": "", "end_date": "", "has_trend": False,
            "line_data": {}, "rolling": {}, "anomalies": [],
            "error": "Invalid columns",
        }

    df_ts = df[[time_col, value_col]].copy()
    try:
        df_ts[time_col] = pd.to_datetime(df_ts[time_col])
    except Exception:
        return {
            "time_col": time_col, "value_col": value_col, "n_points": 0,
            "start_date": "", "end_date": "", "has_trend": False,
            "line_data": {}, "rolling": {}, "anomalies": [],
            "error": f"Cannot parse {time_col} as datetime",
        }

    df_ts = df_ts.sort_values(time_col).dropna()
    if len(df_ts) < 10:
        return {
            "time_col": time_col, "value_col": value_col, "n_points": len(df_ts),
            "start_date": "", "end_date": "", "has_trend": False,
            "line_data": {}, "rolling": {}, "anomalies": [],
            "error": "Need at least 10 data points for time series analysis",
        }

    df_ts = df_ts.reset_index(drop=True)
    series = df_ts[value_col].astype(float)
    dates = df_ts[time_col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    MAX_POINTS = 2000
    dates_trunc = dates[:MAX_POINTS]
    values_trunc = [_safe(v) for v in series.values[:MAX_POINTS]]

    result: dict = {
        "time_col": time_col,
        "value_col": value_col,
        "n_points": len(series),
        "start_date": dates[0],
        "end_date": dates[-1],
        "has_trend": False,
        "seasonality": None,
        "adf_statistic": None,
        "adf_pvalue": None,
        "is_stationary": None,
        "line_data": {"dates": dates_trunc, "values": values_trunc},
        "rolling": {},
        "decomposition": None,
        "acf": None,
        "pacf": None,
        "anomalies": [],
    }

    # Rolling stats
    window = max(3, len(series) // 20)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    result["rolling"] = {
        "window": window,
        "mean": [_safe(v) for v in rolling_mean.values[:MAX_POINTS]],
        "std": [_safe(v) for v in rolling_std.values[:MAX_POINTS]],
    }

    # ADF test
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_res = adfuller(series.dropna())
        result["adf_statistic"] = _safe(adf_res[0])
        result["adf_pvalue"] = _safe(adf_res[1])
        result["is_stationary"] = bool(float(adf_res[1]) < 0.05)
    except Exception:
        pass

    # Decomposition
    if len(series) >= 24:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            period = min(12, len(series) // 4)
            decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
            result["decomposition"] = {
                "trend": [_safe(v) for v in decomp.trend.values[:MAX_POINTS]],
                "seasonal": [_safe(v) for v in decomp.seasonal.values[:MAX_POINTS]],
                "residual": [_safe(v) for v in decomp.resid.values[:MAX_POINTS]],
                "dates": dates_trunc,
            }
            trend_clean = decomp.trend.dropna()
            if len(trend_clean) > 1:
                result["has_trend"] = bool(abs(float(trend_clean.diff().mean())) > 0)
        except Exception:
            pass

    # ACF / PACF
    try:
        from statsmodels.tsa.stattools import acf, pacf
        n_lags = min(40, len(series) // 3)
        acf_vals = acf(series.dropna(), nlags=n_lags)
        pacf_vals = pacf(series.dropna(), nlags=n_lags)
        result["acf"] = {"values": [_safe(v) for v in acf_vals.tolist()]}
        result["pacf"] = {"values": [_safe(v) for v in pacf_vals.tolist()]}
    except Exception:
        pass

    # Anomaly detection (rolling Z-score)
    try:
        rm = series.rolling(window=window, center=True, min_periods=1).mean()
        rs = series.rolling(window=window, center=True, min_periods=1).std().fillna(1)
        z_scores = abs((series - rm) / (rs + 1e-9))
        anomaly_mask = z_scores > 3
        anomalies = []
        for idx in series.index[anomaly_mask]:
            anomalies.append({
                "index": int(idx),
                "date": df_ts.loc[idx, time_col].strftime("%Y-%m-%dT%H:%M:%S"),
                "value": _safe(series[idx]),
            })
        result["anomalies"] = anomalies[:50]
    except Exception:
        pass

    return result
