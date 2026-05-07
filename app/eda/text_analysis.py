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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def run_text_analysis(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {
            "column": column, "total_texts": 0, "avg_length": 0.0, "median_length": 0.0,
            "word_freq": [], "bigrams": [], "trigrams": [],
            "sentiment_dist": {"positive": 0, "negative": 0, "neutral": 0},
            "language": "en", "length_distribution": {},
            "error": f"Column '{column}' not found",
        }

    series = df[column].dropna().astype(str)
    if len(series) == 0:
        return {
            "column": column, "total_texts": 0, "avg_length": 0.0, "median_length": 0.0,
            "word_freq": [], "bigrams": [], "trigrams": [],
            "sentiment_dist": {"positive": 0, "negative": 0, "neutral": 0},
            "language": "en", "length_distribution": {},
            "error": "No data",
        }

    all_tokens: list[str] = []
    all_bigrams: list[str] = []
    all_trigrams: list[str] = []
    lengths: list[int] = []

    for text in series:
        tokens = _tokenize(text)
        lengths.append(len(tokens))
        filtered = [t for t in tokens if t not in STOPWORDS]
        all_tokens.extend(filtered)
        all_bigrams.extend(_ngrams(filtered, 2))
        all_trigrams.extend(_ngrams(filtered, 3))

    word_freq = [{"word": w, "count": c} for w, c in Counter(all_tokens).most_common(30)]
    bigrams = [{"ngram": ng, "count": c} for ng, c in Counter(all_bigrams).most_common(20)]
    trigrams = [{"ngram": ng, "count": c} for ng, c in Counter(all_trigrams).most_common(20)]

    # Sentiment (VADER with fallback)
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

    # Language detection
    language = "en"
    try:
        from langdetect import detect
        sample_text = " ".join(series.head(10).tolist())
        language = detect(sample_text)
    except Exception:
        pass

    # Length distribution
    lengths_arr = np.array(lengths)
    hist_counts, hist_bins = np.histogram(lengths_arr, bins=min(20, max(5, len(set(lengths)))))

    return {
        "column": column,
        "total_texts": len(series),
        "avg_length": round(float(np.mean(lengths)), 2),
        "median_length": round(float(np.median(lengths)), 2),
        "word_freq": word_freq,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "sentiment_dist": sentiment_dist,
        "language": language,
        "length_distribution": {
            "bins": [round(float(x), 1) for x in hist_bins.tolist()],
            "counts": hist_counts.tolist(),
        },
    }
