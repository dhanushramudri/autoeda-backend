from typing import Any, Optional
from pydantic import BaseModel


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    unique_count: int
    unique_pct: float
    missing_count: int
    missing_pct: float
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    top_values: list[dict[str, Any]] = []


class ProfileResult(BaseModel):
    total_rows: int
    total_columns: int
    memory_mb: float
    file_size_bytes: Optional[int] = None
    duplicate_count: int
    duplicate_pct: float
    sampled: bool = False
    sample_size: int = 0
    columns: list[ColumnProfile]


class MissingResult(BaseModel):
    columns: list[dict[str, Any]]
    total_missing: int
    missing_pct: float
    correlation_matrix: dict[str, Any]
    mcar_indicators: dict[str, Any]
    imputation_suggestions: dict[str, str]


class DistributionResult(BaseModel):
    column: str
    is_numeric: bool
    histogram: Optional[dict[str, Any]] = None
    kde: Optional[dict[str, Any]] = None
    box_stats: Optional[dict[str, Any]] = None
    qq_plot: Optional[dict[str, Any]] = None
    normality: Optional[dict[str, Any]] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    bar_chart: Optional[dict[str, Any]] = None
    unique_count: Optional[int] = None
    top_category: Optional[str] = None
    error: Optional[str] = None


class CorrelationResult(BaseModel):
    method: str
    matrix: dict[str, Any]
    top_pairs: list[dict[str, Any]]
    vif: Optional[list[dict[str, Any]]] = None
    cramers_v: Optional[dict[str, Any]] = None


class OutlierResult(BaseModel):
    method: str
    columns: list[dict[str, Any]]
    outlier_rows: list[dict[str, Any]]
    total_outliers: int


class FeatureImportanceResult(BaseModel):
    target: str
    problem_type: str
    importances: list[dict[str, Any]]
    mutual_info: list[dict[str, Any]]
    correlations: list[dict[str, Any]]
    error: Optional[str] = None


class TimeSeriesResult(BaseModel):
    time_col: str
    value_col: str
    n_points: int
    start_date: str
    end_date: str
    has_trend: bool
    seasonality: Optional[str] = None
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None
    is_stationary: Optional[bool] = None
    line_data: dict[str, Any]
    rolling: dict[str, Any]
    decomposition: Optional[dict[str, Any]] = None
    acf: Optional[dict[str, Any]] = None
    pacf: Optional[dict[str, Any]] = None
    anomalies: list[dict[str, Any]] = []
    error: Optional[str] = None


class TextResult(BaseModel):
    column: str
    total_texts: int
    avg_length: float
    median_length: float
    word_freq: list[dict[str, Any]]
    bigrams: list[dict[str, Any]]
    trigrams: list[dict[str, Any]]
    sentiment_dist: dict[str, int]
    language: str
    length_distribution: dict[str, Any]
    error: Optional[str] = None


class QualityScore(BaseModel):
    overall: int
    completeness: int
    consistency: int
    uniqueness: int
    validity: int
    issues: list[dict[str, Any]]
    suggestions: list[str]


class InsightCard(BaseModel):
    chart_type: str
    insight: str
    severity: str  # info | warning | danger


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: Optional[str] = None
    result_data: Optional[dict[str, Any]] = None
