import re
from collections import Counter

import numpy as np
import pandas as pd

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by","is",
    "it","this","that","was","are","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","need","i","you","he",
    "she","we","they","me","him","her","us","them","my","your","his","our","their",
    "its","what","which","who","not","no","as","from","up","out","if","then","than",
    "so","also","just","about","into","through","during","more","very","some","any",
}

SAMPLE_CAP = 20_000

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"https?://[^\s,]+|www\.[^\s,]+", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _empty_result(column: str, error: str, total_rows: int = 0) -> dict:
    return {
        "column": column, "total_rows": total_rows, "total_texts": 0,
        "missing_count": 0, "missing_pct": 0.0,
        "empty_count": 0, "empty_pct": 0.0,
        "duplicate_count": 0, "duplicate_pct": 0.0, "top_duplicates": [],
        "avg_length": 0.0, "median_length": 0.0,
        "avg_char_length": 0.0, "median_char_length": 0.0,
        "min_char_length": 0, "max_char_length": 0,
        "vocabulary_size": 0, "type_token_ratio": 0.0,
        "word_freq": [], "tfidf_keywords": [], "bigrams": [], "trigrams": [],
        "sentiment_dist": {"positive": 0, "negative": 0, "neutral": 0},
        "language": "en",
        "length_distribution": {}, "char_length_distribution": {},
        "quality_flags": {
            "outlier_short_count": 0, "outlier_short_pct": 0.0,
            "outlier_long_count": 0, "outlier_long_pct": 0.0,
            "all_caps_count": 0, "all_caps_pct": 0.0,
            "numeric_only_count": 0, "numeric_only_pct": 0.0,
            "avg_special_char_ratio": 0.0,
        },
        "pii": {
            "emails": {"count": 0, "unique_count": 0, "samples": []},
            "urls": {"count": 0, "unique_count": 0, "samples": []},
            "phone_numbers": {"count": 0, "unique_count": 0, "samples": []},
        },
        "insights": [],
        "sampled": False,
        "sample_size": None,
        "error": error,
    }


def run_text_analysis(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return _empty_result(column, f"Column '{column}' not found")

    total_rows = len(df)
    raw = df[column]
    missing_count = int(raw.isna().sum())
    series_full = raw.dropna().astype(str)

    if len(series_full) == 0:
        return _empty_result(column, "No data", total_rows)

    empty_mask = series_full.str.strip() == ""
    empty_count = int(empty_mask.sum())

    n_unique = int(series_full.nunique())
    duplicate_count = len(series_full) - n_unique
    vc = series_full.value_counts()
    top_duplicates = [
        {"text": (t[:200] + "…" if len(t) > 200 else t), "count": int(c)}
        for t, c in vc[vc > 1].head(10).items()
    ]

    char_lengths_full = series_full.str.len()
    q1, q3 = char_lengths_full.quantile(0.25), char_lengths_full.quantile(0.75)
    long_fence = q3 + 1.5 * (q3 - q1)
    outlier_short_count = int((char_lengths_full <= 2).sum())
    outlier_long_count = int((char_lengths_full > long_fence).sum())

    char_hist_counts, char_hist_bins = np.histogram(
        char_lengths_full.values, bins=min(20, max(5, char_lengths_full.nunique() or 1))
    )

    sampled = len(series_full) > SAMPLE_CAP
    series = series_full.sample(SAMPLE_CAP, random_state=42) if sampled else series_full
    n_sampled = len(series)

    all_tokens: list[str] = []
    all_bigrams: list[str] = []
    all_trigrams: list[str] = []
    word_lengths: list[int] = []
    all_caps_count = 0
    numeric_only_count = 0
    special_ratios: list[float] = []

    for text in series:
        tokens = _tokenize(text)
        word_lengths.append(len(tokens))
        filtered = [t for t in tokens if t not in STOPWORDS]
        all_tokens.extend(filtered)
        all_bigrams.extend(_ngrams(filtered, 2))
        all_trigrams.extend(_ngrams(filtered, 3))

        stripped = text.strip()
        if stripped and stripped.isupper() and any(c.isalpha() for c in stripped):
            all_caps_count += 1
        if stripped and stripped.replace(".", "", 1).replace(",", "", 1).isdigit():
            numeric_only_count += 1
        if len(text) > 0:
            special = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_ratios.append(special / len(text))

    word_freq = [{"word": w, "count": c} for w, c in Counter(all_tokens).most_common(30)]
    bigrams = [{"ngram": ng, "count": c} for ng, c in Counter(all_bigrams).most_common(20)]
    trigrams = [{"ngram": ng, "count": c} for ng, c in Counter(all_trigrams).most_common(20)]

    vocabulary_size = len(set(all_tokens))
    type_token_ratio = round(vocabulary_size / len(all_tokens), 4) if all_tokens else 0.0

    tfidf_keywords: list[dict] = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        non_empty = series[series.str.strip() != ""]
        if len(non_empty) >= 2:
            vec = TfidfVectorizer(stop_words="english", max_features=30, min_df=1)
            tfidf = vec.fit_transform(non_empty)
            scores = np.asarray(tfidf.mean(axis=0)).ravel()
            terms = vec.get_feature_names_out()
            ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            tfidf_keywords = [{"word": t, "score": round(float(s), 5)} for t, s in ranked[:20]]
    except Exception:
        pass

    # ── Sentiment (VADER with lexicon fallback) ─────────────────────────────
    sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        for text in series:
            score = analyzer.polarity_scores(text)["compound"]
            if score >= 0.05:
                sentiment_dist["positive"] += 1
            elif score <= -0.05:
                sentiment_dist["negative"] += 1
            else:
                sentiment_dist["neutral"] += 1
    except ImportError:
        positive_words = {"good","great","excellent","positive","best","amazing","wonderful","love","nice","perfect"}
        negative_words = {"bad","poor","terrible","negative","worst","awful","hate","horrible","wrong","fail"}
        for text in series:
            token_set = set(_tokenize(text))
            pos = len(token_set & positive_words)
            neg = len(token_set & negative_words)
            if pos > neg:
                sentiment_dist["positive"] += 1
            elif neg > pos:
                sentiment_dist["negative"] += 1
            else:
                sentiment_dist["neutral"] += 1

    # ── Language detection ───────────────────────────────────────────────
    language = "en"
    try:
        from langdetect import detect

        sample_text = " ".join(series.head(10).tolist())
        language = detect(sample_text)
    except Exception:
        pass

    def _pattern_stats(pattern: re.Pattern) -> dict:
        matches: list[str] = []
        for text in series:
            matches.extend(pattern.findall(text))
        uniq = list(dict.fromkeys(matches))
        return {"count": len(matches), "unique_count": len(uniq), "samples": uniq[:5]}

    pii = {
        "emails": _pattern_stats(EMAIL_RE),
        "urls": _pattern_stats(URL_RE),
        "phone_numbers": _pattern_stats(PHONE_RE),
    }

    quality_flags = {
        "outlier_short_count": outlier_short_count,
        "outlier_short_pct": round(outlier_short_count / len(series_full) * 100, 2),
        "outlier_long_count": outlier_long_count,
        "outlier_long_pct": round(outlier_long_count / len(series_full) * 100, 2),
        "all_caps_count": all_caps_count,
        "all_caps_pct": round(all_caps_count / max(n_sampled, 1) * 100, 2),
        "numeric_only_count": numeric_only_count,
        "numeric_only_pct": round(numeric_only_count / max(n_sampled, 1) * 100, 2),
        "avg_special_char_ratio": round(float(np.mean(special_ratios)), 4) if special_ratios else 0.0,
    }

    word_lengths_arr = np.array(word_lengths)
    hist_counts, hist_bins = np.histogram(
        word_lengths_arr, bins=min(20, max(5, len(set(word_lengths)) or 1))
    )

    missing_pct = round(missing_count / total_rows * 100, 2) if total_rows else 0.0
    empty_pct = round(empty_count / len(series_full) * 100, 2) if len(series_full) else 0.0
    duplicate_pct = round(duplicate_count / len(series_full) * 100, 2) if len(series_full) else 0.0

    insights: list[dict] = []
    if missing_pct > 20:
        insights.append({"type": "high_missing", "level": "warning",
                          "message": f"{missing_pct}% of rows are missing in '{column}'."})
    if empty_pct > 10:
        insights.append({"type": "high_empty", "level": "warning",
                          "message": f"{empty_pct}% of non-null values are blank or whitespace-only."})
    if duplicate_pct > 30:
        insights.append({"type": "high_duplicates", "level": "info",
                          "message": f"{duplicate_pct}% of values are exact duplicates of another row."})
    if pii["emails"]["count"] > 0:
        insights.append({"type": "pii_email", "level": "danger",
                          "message": f"{pii['emails']['count']} email address(es) detected — possible PII."})
    if pii["phone_numbers"]["count"] > 0:
        insights.append({"type": "pii_phone", "level": "danger",
                          "message": f"{pii['phone_numbers']['count']} phone-number-like pattern(s) detected — possible PII."})
    total_sent = sum(sentiment_dist.values()) or 1
    dominant_sent_pct = max(sentiment_dist.values()) / total_sent * 100
    if dominant_sent_pct > 85:
        dominant = max(sentiment_dist, key=sentiment_dist.get)
        insights.append({"type": "sentiment_skew", "level": "info",
                          "message": f"{dominant_sent_pct:.0f}% of texts are classified {dominant} — low sentiment diversity."})
    if type_token_ratio < 0.05 and len(all_tokens) > 200:
        insights.append({"type": "low_vocabulary", "level": "info",
                          "message": f"Low vocabulary richness (type-token ratio {type_token_ratio}) — text may be repetitive or templated."})
    if quality_flags["outlier_long_pct"] > 5:
        insights.append({"type": "long_outliers", "level": "info",
                          "message": f"{quality_flags['outlier_long_pct']}% of values are unusually long relative to the rest of the column."})

    return {
        "column": column,
        "total_rows": total_rows,
        "total_texts": len(series_full),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "empty_count": empty_count,
        "empty_pct": empty_pct,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "top_duplicates": top_duplicates,
        "avg_length": round(float(np.mean(word_lengths)), 2) if word_lengths else 0.0,
        "median_length": round(float(np.median(word_lengths)), 2) if word_lengths else 0.0,
        "avg_char_length": round(float(char_lengths_full.mean()), 2),
        "median_char_length": round(float(char_lengths_full.median()), 2),
        "min_char_length": int(char_lengths_full.min()),
        "max_char_length": int(char_lengths_full.max()),
        "vocabulary_size": vocabulary_size,
        "type_token_ratio": type_token_ratio,
        "word_freq": word_freq,
        "tfidf_keywords": tfidf_keywords,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "sentiment_dist": sentiment_dist,
        "language": language,
        "length_distribution": {
            "bins": [round(float(x), 1) for x in hist_bins.tolist()],
            "counts": hist_counts.tolist(),
        },
        "char_length_distribution": {
            "bins": [round(float(x), 1) for x in char_hist_bins.tolist()],
            "counts": char_hist_counts.tolist(),
        },
        "quality_flags": quality_flags,
        "pii": pii,
        "insights": insights,
        "sampled": sampled,
        "sample_size": n_sampled if sampled else None,
    }
