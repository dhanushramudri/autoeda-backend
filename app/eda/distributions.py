import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde


def _safe(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def run_distribution(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {"column": column, "is_numeric": False, "error": f"Column '{column}' not found"}

    series = df[column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    result: dict = {"column": column, "is_numeric": is_numeric}

    if len(series) == 0:
        result["error"] = "No data after dropping nulls"
        return result

    if is_numeric:
        series_f = series.astype(float)
        n_bins = min(50, max(10, int(np.sqrt(len(series_f)))))
        counts, bin_edges = np.histogram(series_f, bins=n_bins)
        result["histogram"] = {
            "bins": [_safe(x) for x in bin_edges.tolist()],
            "counts": counts.tolist(),
        }

        # KDE
        if len(series_f) > 1 and series_f.std() > 0:
            try:
                kde = gaussian_kde(series_f)
                x_range = np.linspace(series_f.min(), series_f.max(), 200)
                result["kde"] = {
                    "x": [_safe(x) for x in x_range.tolist()],
                    "y": [_safe(y) for y in kde(x_range).tolist()],
                }
            except Exception:
                result["kde"] = None
        else:
            result["kde"] = None

        q1 = _safe(series_f.quantile(0.25))
        median = _safe(series_f.median())
        q3 = _safe(series_f.quantile(0.75))
        iqr = (q3 or 0) - (q1 or 0)
        lower = (q1 or 0) - 1.5 * iqr
        upper = (q3 or 0) + 1.5 * iqr
        outliers_vals = series_f[(series_f < lower) | (series_f > upper)].head(100).tolist()
        result["box_stats"] = {
            "min": _safe(series_f.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": _safe(series_f.max()),
            "outliers": [_safe(v) for v in outliers_vals],
        }

        # QQ plot
        try:
            qq = stats.probplot(series_f, dist="norm")
            theoretical = [_safe(x) for x in qq[0][0][:200].tolist()]
            sample = [_safe(x) for x in qq[0][1][:200].tolist()]
            result["qq_plot"] = {"theoretical": theoretical, "sample": sample}
        except Exception:
            result["qq_plot"] = None

        # Normality test
        try:
            if len(series_f) < 5000:
                stat, p = stats.shapiro(series_f.head(5000))
                test_name = "shapiro"
            else:
                stat, p = stats.normaltest(series_f)
                test_name = "dagostino"
            result["normality"] = {
                "test": test_name,
                "statistic": _safe(stat),
                "p_value": _safe(p),
                "is_normal": bool(float(p) > 0.05),
            }
        except Exception:
            result["normality"] = None

        result["skewness"] = _safe(series_f.skew())
        result["kurtosis"] = _safe(series_f.kurtosis())

    else:
        vc = series.value_counts().head(20)
        result["bar_chart"] = {
            "labels": [str(v) for v in vc.index.tolist()],
            "counts": vc.values.tolist(),
        }
        result["unique_count"] = int(series.nunique())
        result["top_category"] = str(series.mode().iloc[0]) if len(series.mode()) > 0 else None

    return result
