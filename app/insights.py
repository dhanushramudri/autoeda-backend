class InsightEngine:
    def from_profile(self, profile: dict) -> list[dict]:
        insights = []
        for col in profile.get("columns", []):
            name = col["name"]
            skew = col.get("skewness")
            missing_pct = col.get("missing_pct", 0)
            sem_type = col.get("semantic_type", "")

            if missing_pct > 50:
                insights.append({
                    "chart_type": "Data Profile",
                    "insight": f"'{name}' has {missing_pct:.1f}% missing values — consider dropping before modeling.",
                    "severity": "danger",
                })
            elif missing_pct > 20:
                insights.append({
                    "chart_type": "Data Profile",
                    "insight": f"'{name}' has {missing_pct:.1f}% missing values. Imputation required.",
                    "severity": "warning",
                })

            if skew is not None:
                if skew > 2:
                    insights.append({
                        "chart_type": "Distribution",
                        "insight": f"'{name}' is highly right-skewed (skewness={skew:.2f}). Consider log transformation before modeling.",
                        "severity": "warning",
                    })
                elif skew < -2:
                    insights.append({
                        "chart_type": "Distribution",
                        "insight": f"'{name}' is highly left-skewed (skewness={skew:.2f}). Consider square root transformation.",
                        "severity": "warning",
                    })

            if sem_type == "constant":
                insights.append({
                    "chart_type": "Data Profile",
                    "insight": f"'{name}' has zero variance (constant value) and should be removed before modeling.",
                    "severity": "warning",
                })
            elif sem_type == "id_like":
                insights.append({
                    "chart_type": "Data Profile",
                    "insight": f"'{name}' appears to be an identifier column (very high cardinality). Drop before ML modeling.",
                    "severity": "info",
                })

        if profile.get("duplicate_pct", 0) > 5:
            insights.append({
                "chart_type": "Data Quality",
                "insight": f"Dataset contains {profile['duplicate_count']} duplicate rows ({profile['duplicate_pct']:.1f}%). Clean before analysis.",
                "severity": "warning",
            })

        return insights

    def from_distribution(self, dist: dict) -> list[dict]:
        insights = []
        col = dist.get("column", "")
        norm = dist.get("normality")
        skew = dist.get("skewness")

        if norm:
            p = norm.get("p_value")
            if p is not None:
                if p > 0.05:
                    insights.append({
                        "chart_type": "Distribution",
                        "insight": f"'{col}' follows a normal distribution (p={p:.4f}). Parametric tests are appropriate.",
                        "severity": "info",
                    })
                else:
                    insights.append({
                        "chart_type": "Distribution",
                        "insight": f"'{col}' does not follow a normal distribution (p={p:.4f}). Use non-parametric tests.",
                        "severity": "warning",
                    })

        if skew is not None:
            if abs(skew) > 2:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "chart_type": "Distribution",
                    "insight": f"'{col}' is highly {direction}-skewed (skewness={skew:.2f}). Log or Box-Cox transform recommended.",
                    "severity": "warning",
                })

        return insights

    def from_correlations(self, corr: dict) -> list[dict]:
        insights = []
        for pair in corr.get("top_pairs", [])[:5]:
            r = pair.get("correlation", 0)
            if abs(r) > 0.9:
                insights.append({
                    "chart_type": "Correlation",
                    "insight": f"'{pair['col1']}' and '{pair['col2']}' are highly correlated (r={r:.2f}). Consider removing one to reduce multicollinearity.",
                    "severity": "warning",
                })
            elif abs(r) > 0.7:
                insights.append({
                    "chart_type": "Correlation",
                    "insight": f"'{pair['col1']}' and '{pair['col2']}' have strong correlation (r={r:.2f}).",
                    "severity": "info",
                })

        for vif_item in (corr.get("vif") or [])[:3]:
            vif_val = vif_item.get("vif", 0)
            if vif_val > 10:
                insights.append({
                    "chart_type": "Multicollinearity",
                    "insight": f"'{vif_item['column']}' has high VIF ({vif_val:.1f}) — strong multicollinearity detected.",
                    "severity": "danger",
                })
            elif vif_val > 5:
                insights.append({
                    "chart_type": "Multicollinearity",
                    "insight": f"'{vif_item['column']}' has moderate VIF ({vif_val:.1f}) — some multicollinearity present.",
                    "severity": "warning",
                })

        return insights

    def from_outliers(self, outlier_result: dict) -> list[dict]:
        insights = []
        for col_info in outlier_result.get("columns", []):
            pct = col_info.get("outlier_pct", 0)
            name = col_info.get("name", "")
            if pct > 10:
                insights.append({
                    "chart_type": "Outliers",
                    "insight": f"'{name}' has {pct:.1f}% outliers by {outlier_result.get('method','IQR')} method. Review before modeling.",
                    "severity": "warning",
                })
            elif pct > 3:
                insights.append({
                    "chart_type": "Outliers",
                    "insight": f"'{name}' has {pct:.1f}% outliers detected. Consider outlier treatment.",
                    "severity": "info",
                })
        return insights

    def from_quality_score(self, quality: dict) -> list[dict]:
        insights = []
        score = quality.get("overall", 100)
        if score < 60:
            insights.append({
                "chart_type": "Data Quality",
                "insight": f"Data Quality Score is low ({score}/100). Address critical issues before analysis.",
                "severity": "danger",
            })
        elif score < 80:
            insights.append({
                "chart_type": "Data Quality",
                "insight": f"Data Quality Score is moderate ({score}/100). Some improvements recommended.",
                "severity": "warning",
            })
        else:
            insights.append({
                "chart_type": "Data Quality",
                "insight": f"Data Quality Score is good ({score}/100). Dataset is ready for analysis.",
                "severity": "info",
            })
        return insights

    def from_timeseries(self, ts: dict) -> list[dict]:
        insights = []
        if ts.get("is_stationary") is False:
            p = ts.get("adf_pvalue")
            insights.append({
                "chart_type": "Time Series",
                "insight": f"Time series is non-stationary (ADF p={p:.4f if p else 'N/A'}). Consider differencing before ARIMA modeling.",
                "severity": "warning",
            })
        elif ts.get("is_stationary"):
            insights.append({
                "chart_type": "Time Series",
                "insight": "Time series is stationary (ADF test passed). Suitable for direct modeling.",
                "severity": "info",
            })
        n_anomalies = len(ts.get("anomalies", []))
        if n_anomalies > 0:
            insights.append({
                "chart_type": "Time Series",
                "insight": f"Detected {n_anomalies} anomalies in the time series using rolling Z-score method.",
                "severity": "warning",
            })
        return insights

    def from_missing(self, missing: dict) -> list[dict]:
        insights = []
        total_missing_pct = missing.get("missing_pct", 0)
        if total_missing_pct > 30:
            insights.append({
                "chart_type": "Missing Values",
                "insight": f"Overall dataset missing rate is {total_missing_pct:.1f}% — consider imputation or dropping high-missing columns before modeling.",
                "severity": "danger",
            })
        elif total_missing_pct > 10:
            insights.append({
                "chart_type": "Missing Values",
                "insight": f"Dataset has {total_missing_pct:.1f}% missing values overall. Review imputation strategies.",
                "severity": "warning",
            })

        for col in missing.get("columns", []):
            pct = col.get("pct", 0)
            name = col.get("name", "")
            suggestion = missing.get("imputation_suggestions", {}).get(name, "")
            mcar = missing.get("mcar_indicators", {}).get(name, {})

            if pct > 80:
                insights.append({
                    "chart_type": "Missing Values",
                    "insight": f"'{name}' has {pct:.1f}% missing — drop this column before modeling.",
                    "severity": "danger",
                })
            elif pct > 40:
                msg = f"'{name}' has {pct:.1f}% missing."
                if suggestion:
                    msg += f" Suggested strategy: {suggestion}."
                insights.append({"chart_type": "Missing Values", "insight": msg, "severity": "warning"})

            if mcar.get("likely") == "MCAR":
                corr_with = mcar.get("correlated_with", [])
                if corr_with and pct > 5:
                    insights.append({
                        "chart_type": "Missing Values",
                        "insight": f"'{name}' missingness correlates with {', '.join(corr_with[:2])} — possibly Not Missing At Random (NMAR). Treat carefully.",
                        "severity": "warning",
                    })
        return insights

    def from_feature_importance(self, fi: dict) -> list[dict]:
        insights = []
        target = fi.get("target", "target")
        importances = fi.get("importances", [])
        problem_type = fi.get("problem_type", "")

        if not importances:
            if fi.get("error"):
                insights.append({
                    "chart_type": "Feature Importance",
                    "insight": f"Feature importance could not be computed: {fi['error']}",
                    "severity": "warning",
                })
            return insights

        top = importances[0] if importances else None
        if top:
            insights.append({
                "chart_type": "Feature Importance",
                "insight": f"'{top.get('feature')}' is the most important feature for predicting '{target}' (importance={top.get('importance', 0):.3f}).",
                "severity": "info",
            })

        low_importance = [f for f in importances if f.get("importance", 1) < 0.01]
        if len(low_importance) > 3:
            names = ", ".join(f["feature"] for f in low_importance[:5])
            insights.append({
                "chart_type": "Feature Importance",
                "insight": f"{len(low_importance)} features have near-zero importance: {names}. Consider removing to reduce model complexity.",
                "severity": "info",
            })

        mutual_info = fi.get("mutual_info", [])
        if mutual_info and mutual_info[0].get("score", 0) > 0.5:
            top_mi = mutual_info[0]
            insights.append({
                "chart_type": "Feature Importance",
                "insight": f"'{top_mi['feature']}' has high mutual information with '{target}' (MI={top_mi['score']:.3f}), confirming predictive power.",
                "severity": "info",
            })

        for c in fi.get("correlations", [])[:3]:
            r = abs(c.get("correlation", 0))
            if r > 0.95:
                insights.append({
                    "chart_type": "Feature Importance",
                    "insight": f"'{c['feature']}' has near-perfect correlation with '{target}' (r={c['correlation']:.3f}) — possible target leakage, investigate before training.",
                    "severity": "danger",
                })

        insights.append({
            "chart_type": "Feature Importance",
            "insight": f"Problem type detected: '{problem_type}'. Using {'Random Forest' if problem_type in ('regression', 'classification') else 'heuristic'} for importance ranking.",
            "severity": "info",
        })
        return insights

    def from_text_analysis(self, text: dict) -> list[dict]:
        insights = []
        total = text.get("total_texts", 0)
        if total == 0:
            return insights

        sentiment_dist = text.get("sentiment_dist", {})
        negative_pct = (sentiment_dist.get("negative", 0) / total * 100) if total > 0 else 0
        positive_pct = (sentiment_dist.get("positive", 0) / total * 100) if total > 0 else 0
        if negative_pct > 50:
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"{negative_pct:.1f}% of texts have negative sentiment. Investigate underlying issues.",
                "severity": "warning",
            })
        elif positive_pct > 80:
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"{positive_pct:.1f}% of texts have positive sentiment — data may be skewed or filtered.",
                "severity": "info",
            })

        avg_len = text.get("avg_length", 0)
        if avg_len < 10:
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"Average text length is very short ({avg_len:.0f} chars). NLP models may struggle with such brief inputs.",
                "severity": "warning",
            })
        elif avg_len > 5000:
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"Average text length is very long ({avg_len:.0f} chars). Consider chunking for NLP processing.",
                "severity": "info",
            })

        language = text.get("language", "en")
        if language != "en":
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"Detected non-English language ('{language}'). Ensure NLP models support this language.",
                "severity": "info",
            })

        word_freq = text.get("word_freq", [])
        if word_freq and word_freq[0].get("count", 0) > total * 0.9:
            insights.append({
                "chart_type": "Text Analysis",
                "insight": f"Top word '{word_freq[0]['word']}' appears in {word_freq[0]['count']/total*100:.0f}% of texts — may be a stop word worth filtering.",
                "severity": "info",
            })

        return insights