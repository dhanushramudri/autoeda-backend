from typing import Any, Optional
from pydantic import BaseModel, ConfigDict



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



class CorrelationColumnProfile(BaseModel):
    num_cols:     list[str] = []
    cat_cols:     list[str] = []
    ignored_cols: list[str] = []
    skipped_cols:    list[str]       = []               
    cat_cardinality: dict[str, int]  = {}    


class CatPair(BaseModel):
    col1:      str
    col2:      str
    cramers_v: float


class MixedCell(BaseModel):
    eta_sq:         Optional[float] = None
    point_biserial: Optional[float] = None
    rank_biserial:  Optional[float] = None
    p_value:        Optional[float] = None
    n_categories:   int = 0


class MixedPair(BaseModel):
    num_col:        str
    cat_col:        str
    eta_sq:         Optional[float] = None
    point_biserial: Optional[float] = None
    rank_biserial:  Optional[float] = None
    p_value:        Optional[float] = None
    n_categories:   int = 0


class CorrelationResult(BaseModel):
    method:         str
    column_profile: Optional[CorrelationColumnProfile] = None

    matrix:    dict[str, Any]            = {}
    p_values:  dict[str, Any]            = {}
    top_pairs: list[dict[str, Any]]      = []
    vif:       Optional[list[dict[str, Any]]] = None

    cramers_v:     Optional[dict[str, Any]] = None
    theils_u:      Optional[dict[str, Any]] = None
    cat_p_values:  Optional[dict[str, Any]] = None
    cat_top_pairs: Optional[list[CatPair]] = None

    mixed:           Optional[dict[str, Any]] = None
    mixed_top_pairs: Optional[list[MixedPair]] = None

    insights: Optional[list[dict[str, Any]]] = None


class OutlierResult(BaseModel):
    method: str
    columns: list[dict[str, Any]]
    outlier_rows: list[dict[str, Any]]
    total_outliers: int


class FeatureImportanceResult(BaseModel):
    target: str
    problem_type: str
    n_samples: int = 0
    n_features: int = 0
    model_score: Optional[float] = None
    cv_score_mean: Optional[float] = None        
    cv_score_std: Optional[float] = None         
    class_distribution: Optional[dict[str, Any]] = None
    importances: list[dict[str, Any]] = []
    permutation_importances: list[dict[str, Any]] = []   
    mutual_info: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    anova: list[dict[str, Any]] = []
    shap_values: list[dict[str, Any]] = []      
    feature_meta: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []        
    interactions: list[dict[str, Any]] = []      
    redundant_groups: list[dict[str, Any]] = []  
    leakage_suspects: list[dict[str, Any]] = [] 
    top_features: list[str] = []
    drop_candidates: list[str] = []
    warnings: list[dict[str, Any]] = []
    computed_methods: list[str] = []             
    error: Optional[str] = None

class TimeSeriesResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    time_col: str
    value_col: str
    n_points: int
    start_date: str
    end_date: str

    has_trend: bool

    seasonality: int | None = None

    adf_statistic: float | None = None
    adf_pvalue: float | None = None
    is_stationary: bool | None = None

    line_data: dict[str, Any]
    rolling: dict[str, Any]

    decomposition: dict[str, Any] | None = None
    acf: dict[str, Any] | None = None
    pacf: dict[str, Any] | None = None

    anomalies: list[dict[str, Any]] = []

    error: str | None = None


class TextResult(BaseModel):
    column: str
    total_rows: int
    total_texts: int
    missing_count: int
    missing_pct: float
    empty_count: int
    empty_pct: float
    duplicate_count: int
    duplicate_pct: float
    top_duplicates: list[dict[str, Any]]
    avg_length: float
    median_length: float
    avg_char_length: float
    median_char_length: float
    min_char_length: int
    max_char_length: int
    vocabulary_size: int
    type_token_ratio: float
    word_freq: list[dict[str, Any]]
    tfidf_keywords: list[dict[str, Any]]
    bigrams: list[dict[str, Any]]
    trigrams: list[dict[str, Any]]
    sentiment_dist: dict[str, int]
    language: str
    length_distribution: dict[str, Any]
    char_length_distribution: dict[str, Any]
    quality_flags: dict[str, Any]
    pii: dict[str, Any]
    insights: list[dict[str, Any]]
    sampled: bool
    sample_size: Optional[int] = None
    error: Optional[str] = None


class QualityScore(BaseModel):
    overall: int
    completeness: int
    consistency: int
    uniqueness: int
    issues: list[dict[str, Any]]
    suggestions: list[str]


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: Optional[str] = None
    result_data: Optional[dict[str, Any]] = None


# ─── Comprehensive Analysis Result ────────────────────────────────────────────
class HistogramKDE(BaseModel):
    bins: list[Optional[float]] = []
    counts: list[int] = []
    kde_x: list[Optional[float]] = []
    kde_y: list[Optional[float]] = []
    mean: Optional[float] = None
    median: Optional[float] = None


class BoxStats(BaseModel):
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    outliers: list[Optional[float]] = []


class ViolinData(BaseModel):
    y: list[Optional[float]] = []


class QQData(BaseModel):
    theoretical: list[Optional[float]] = []
    sample: list[Optional[float]] = []
    line_x: list[float] = []
    line_y: list[float] = []


class ECDFData(BaseModel):
    x: list[Optional[float]] = []
    y: list[Optional[float]] = []


class NormalityResult(BaseModel):
    test: str = ""
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    is_normal: Optional[bool] = None


class NumericChartsType(BaseModel):
    histogram_kde: HistogramKDE = HistogramKDE()
    box: BoxStats = BoxStats()
    violin: ViolinData = ViolinData()
    qq: QQData = QQData()
    ecdf: ECDFData = ECDFData()
    normality: NormalityResult = NormalityResult()
    std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class BarChartData(BaseModel):
    labels: list[str] = []
    values: list[int] = []
    percentages: list[float] = []
    other_count: int = 0
    total_categories: int = 0


class PieData(BaseModel):
    labels: list[str] = []
    values: list[int] = []
    percentages: list[float] = []


class ParetoData(BaseModel):
    labels: list[str] = []
    values: list[int] = []
    cumulative_pct: list[float] = []


class CategoricalChartsType(BaseModel):
    bar: BarChartData = BarChartData()
    pie: Optional[PieData] = None
    pareto: ParetoData = ParetoData()


class TimeseriesData(BaseModel):
    dates: list[str] = []
    values: list[Optional[float]] = []


class SeasonalityData(BaseModel):
    by_hour: dict[str, Any] = {}
    by_dow: dict[str, Any] = {}
    by_month: dict[str, Any] = {}


class DatetimeChartsType(BaseModel):
    timeseries: TimeseriesData = TimeseriesData()
    seasonality: SeasonalityData = SeasonalityData()


class ScatterPairType(BaseModel):
    col1: str
    col2: str
    pearson_r: Optional[float] = None
    r2: Optional[float] = None
    x: list[Optional[float]] = []
    y: list[Optional[float]] = []
    line_x: list[float] = []
    line_y: list[float] = []


class GroupedBoxGroup(BaseModel):
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    outliers: list[Optional[float]] = []
    n: int = 0


class GroupedBoxData(BaseModel):
    numeric_col: str
    categorical_col: str
    groups: dict[str, GroupedBoxGroup] = {}


class CorrelationHeatmapData(BaseModel):
    labels: list[str] = []
    z: list[list[Optional[float]]] = []


class MultiColumnAnalysis(BaseModel):
    correlation: CorrelationHeatmapData = CorrelationHeatmapData()
    scatter_pairs: list[ScatterPairType] = []
    grouped_box: Optional[GroupedBoxData] = None


class MissingBarItem(BaseModel):
    column: str
    missing_count: int
    missing_pct: Optional[float] = None


class NormalityRow(BaseModel):
    column: str
    n: int
    test: str
    p_value: Optional[float] = None
    is_normal: Optional[bool] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class OutlierSummaryRow(BaseModel):
    column: str
    outlier_count: int
    outlier_pct: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class CardinalityRow(BaseModel):
    column: str
    unique_count: int
    unique_pct: Optional[float] = None
    flag: str = "normal"  # id_like | constant | binary | low_cardinality | normal
    dtype: str


class DuplicateInfo(BaseModel):
    total_rows: int
    duplicate_count: int
    duplicate_pct: Optional[float] = None


class StatCards(BaseModel):
    normality_table: list[NormalityRow] = []
    outlier_summary: list[OutlierSummaryRow] = []
    cardinality: list[CardinalityRow] = []
    duplicates: DuplicateInfo = DuplicateInfo(total_rows=0, duplicate_count=0)
    missing_bar: list[MissingBarItem] = []


class FullAnalysisResult(BaseModel):
    sampled: bool = False
    sample_size: int = 0
    total_rows: int = 0
    column_types: dict[str, str] = {}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    numeric_charts: dict[str, NumericChartsType] = {}
    categorical_charts: dict[str, CategoricalChartsType] = {}
    datetime_charts: dict[str, DatetimeChartsType] = {}
    multi_column: MultiColumnAnalysis = MultiColumnAnalysis()
    missing_charts: dict[str, Any] = {}
    stat_cards: StatCards = StatCards()
