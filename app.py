from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import streamlit as st

from backend import (
    APP_PALETTE,
    DisasterSeverityPredictor,
    EVENT_COUNT_COLOR_SCALE,
    MLRSeverityPredictor,
    SEVERITY_COLOR_SCALE,
    SEVERITY_MAX,
    apply_chart_theme,
    build_component_chart,
    build_confidence_badge,
    build_country_summary_chart,
    build_feature_importance_chart,
    build_map_figure,
    build_severity_distribution_chart,
    build_severity_gauge,
    build_yearly_events_chart,
    filter_historical_data,
    format_compact_number,
    get_legacy_model_summary,
    load_and_prepare_disaster_data,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "emdat_data.csv"
LEGACY_MODEL_PATH = BASE_DIR / "mlr_model.pkl"
CACHE_VERSION = "severity-scale-v4"          # bumped: temporal holdout + bidirectional fusion
RF_CACHE_VERSION = "rf-temporal-holdout-v1"  # bumped: forces rebuild of cached predictor
RF_CACHE_PATH = BASE_DIR / "rf_predictor_cache.joblib"
RF_CACHE_META_PATH = BASE_DIR / "rf_predictor_cache.meta.json"


@st.cache_data(show_spinner=False)
def get_historical_data(csv_path: str, _cache_version: str = CACHE_VERSION):
    return load_and_prepare_disaster_data(csv_path)


@st.cache_resource(show_spinner=False)
def get_predictor(csv_path: str, _cache_version: str = CACHE_VERSION):
    return MLRSeverityPredictor(load_and_prepare_disaster_data(csv_path))


@st.cache_data(show_spinner=False)
def get_legacy_summary(model_path: str, _cache_version: str = CACHE_VERSION) -> Dict[str, str]:
    return get_legacy_model_summary(model_path)


def _rf_cache_signature(csv_path: str) -> Dict[str, object]:
    csv_file = Path(csv_path)
    stat = csv_file.stat()
    return {
        "cache_version": CACHE_VERSION,
        "rf_cache_version": RF_CACHE_VERSION,
        "csv_name": csv_file.name,
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
    }


def has_cached_rf_predictor(csv_path: str) -> bool:
    if not RF_CACHE_PATH.exists() or not RF_CACHE_META_PATH.exists():
        return False
    try:
        metadata = json.loads(RF_CACHE_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata == _rf_cache_signature(csv_path)


@st.cache_resource(show_spinner=False)
def get_rf_predictor(csv_path: str, _cache_version: str = CACHE_VERSION):
    if has_cached_rf_predictor(csv_path):
        try:
            return joblib.load(RF_CACHE_PATH)
        except Exception:
            pass

    predictor = DisasterSeverityPredictor(load_and_prepare_disaster_data(csv_path))
    try:
        joblib.dump(predictor, RF_CACHE_PATH)
        RF_CACHE_META_PATH.write_text(
            json.dumps(_rf_cache_signature(csv_path), indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return predictor


def inject_theme_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {APP_PALETTE["background"]};
            color: {APP_PALETTE["text"]};
        }}
        .stApp ::selection,
        .stApp *::selection {{
            background: #86b7ff;
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
        }}
        .stApp ::-moz-selection,
        .stApp *::-moz-selection {{
            background: #86b7ff;
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main span,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main div,
        [data-testid="stAppViewContainer"] .main li,
        [data-testid="stAppViewContainer"] .main h1,
        [data-testid="stAppViewContainer"] .main h2,
        [data-testid="stAppViewContainer"] .main h3,
        [data-testid="stAppViewContainer"] .main h4,
        [data-testid="stAppViewContainer"] .main h5,
        [data-testid="stAppViewContainer"] .main h6 {{
            color: {APP_PALETTE["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {APP_PALETTE["sidebar_start"]} 0%, {APP_PALETTE["sidebar_end"]} 100%);
            border-right: 1px solid {APP_PALETTE["border"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {APP_PALETTE["text"]};
        }}
        [data-testid="stSidebar"] div[data-testid="metric-container"] {{
            background: {APP_PALETTE["surface"]};
            border: 1px solid {APP_PALETTE["border"]};
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }}
        [data-testid="stSidebar"] [data-testid="metric-container"] *,
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
        [data-testid="stSidebar"] [data-testid="stMetricValue"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
            opacity: 1 !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] p {{
            color: {APP_PALETTE["muted"]};
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stWidgetLabel"] *,
        [data-testid="stAppViewContainer"] .main .stSlider label,
        [data-testid="stAppViewContainer"] .main .stSelectbox label,
        [data-testid="stAppViewContainer"] .main .stMultiSelect label,
        [data-testid="stAppViewContainer"] .main .stNumberInput label,
        [data-testid="stAppViewContainer"] .main .stFileUploader label {{
            color: {APP_PALETTE["text"]} !important;
            font-weight: 600;
        }}
        div[data-testid="metric-container"] {{
            background: rgba(255, 253, 248, 0.98);
            border: 1px solid {APP_PALETTE["border"]};
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(38, 70, 83, 0.08);
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="metric-container"] *,
        [data-testid="stAppViewContainer"] .main [data-testid="stMetricLabel"],
        [data-testid="stAppViewContainer"] .main [data-testid="stMetricValue"],
        [data-testid="stAppViewContainer"] .main [data-testid="stMetricLabel"] *,
        [data-testid="stAppViewContainer"] .main [data-testid="stMetricValue"] * {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
            opacity: 1 !important;
        }}
        [data-testid="stAppViewContainer"] .main input,
        [data-testid="stAppViewContainer"] .main textarea,
        [data-testid="stAppViewContainer"] .main [data-baseweb="input"] input,
        [data-testid="stAppViewContainer"] .main [data-baseweb="select"] input {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] .main [data-baseweb="input"] > div,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] {{
            background: {APP_PALETTE["surface"]} !important;
            border-color: {APP_PALETTE["border"]} !important;
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stSelectbox"] > div > div,
        [data-testid="stAppViewContainer"] .main [data-testid="stNumberInput"] > div,
        [data-testid="stAppViewContainer"] .main [data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] .main [data-baseweb="input"] > div,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] {{
            background: {APP_PALETTE["surface"]} !important;
            color: {APP_PALETTE["text"]} !important;
            border: 1px solid {APP_PALETTE["border"]} !important;
            box-shadow: none !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stSelectbox"] *,
        [data-testid="stAppViewContainer"] .main [data-testid="stNumberInput"] *,
        [data-testid="stAppViewContainer"] .main [data-baseweb="select"] *,
        [data-testid="stAppViewContainer"] .main [data-baseweb="input"] *,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] button * {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-baseweb="select"] svg,
        [data-testid="stAppViewContainer"] .main [data-baseweb="slider"] * {{
            color: {APP_PALETTE["text"]} !important;
            fill: {APP_PALETTE["text"]} !important;
        }}
        [data-baseweb="popover"],
        [role="listbox"] {{
            background: {APP_PALETTE["surface"]} !important;
            color: {APP_PALETTE["text"]} !important;
            border: 1px solid {APP_PALETTE["border"]} !important;
        }}
        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [role="option"] * {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stAppViewContainer"] .main [data-testid="stFileUploaderDropzoneInstructions"] {{
            color: {APP_PALETTE["muted"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main button[kind="secondary"],
        [data-testid="stAppViewContainer"] .main [data-testid="baseButton-secondary"] {{
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-baseweb="tab"] {{
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main button[role="tab"],
        [data-testid="stAppViewContainer"] .main [data-baseweb="tab"] {{
            background: {APP_PALETTE["surface"]} !important;
            border: 1px solid {APP_PALETTE["border"]} !important;
            border-radius: 12px 12px 0 0 !important;
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main button[role="tab"][aria-selected="true"] {{
            background: {APP_PALETTE["accent_soft"]} !important;
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main button[role="tab"] *,
        [data-testid="stAppViewContainer"] .main [data-baseweb="tab"] * {{
            color: {APP_PALETTE["text"]} !important;
            -webkit-text-fill-color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main details summary,
        [data-testid="stAppViewContainer"] .main details summary * {{
            color: {APP_PALETTE["text"]} !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stPlotlyChart"],
        [data-testid="stAppViewContainer"] .main [data-testid="stDataFrame"] {{
            background: {APP_PALETTE["chart_surface"]};
            border: 1px solid {APP_PALETTE["border"]};
            border-radius: 18px;
            padding: 0.35rem;
        }}
        [data-testid="stAppViewContainer"] .main a {{
            color: {APP_PALETTE["high"]};
        }}
        .stButton > button {{
            background: linear-gradient(90deg, {APP_PALETTE["accent"]} 0%, {APP_PALETTE["storm"]} 100%);
            color: #1f2933;
            border: 0;
            border-radius: 999px;
            font-weight: 700;
        }}
        .stAlert {{
            border-radius: 14px;
            border: 1px solid {APP_PALETTE["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_prediction_result(result: Dict) -> Dict:
    if not result:
        return result

    normalized = dict(result)
    hybrid_score = float(normalized.get("hybrid_score", 0.0))

    # Old cached results were stored on the 0-100 scale. Convert them so the UI
    # doesn't break while Streamlit rebuilds the cached predictors.
    if hybrid_score > SEVERITY_MAX:
        normalized["hybrid_score"] = round(hybrid_score / 10.0, 1)

        prediction_range = normalized.get("prediction_range", {})
        if prediction_range:
            normalized["prediction_range"] = {
                "lower": round(float(prediction_range.get("lower", 0.0)) / 10.0, 1),
                "upper": round(float(prediction_range.get("upper", 0.0)) / 10.0, 1),
            }

        for score_key in ("linear_score", "base_rf_score", "ratio_rf_score", "image_only_damage_score"):
            if normalized.get(score_key) is not None:
                normalized[score_key] = round(float(normalized[score_key]) / 10.0, 1)

        component_scores = normalized.get("component_scores", {})
        if component_scores:
            normalized["component_scores"] = {
                name: round(float(score) / 10.0, 1) for name, score in component_scores.items()
            }

    normalized.setdefault("image_ratio_used", float(normalized.get("image_damage_ratio") or 0.0))
    normalized.setdefault(
        "image_ratio_source",
        "Uploaded image" if normalized.get("image_damage_ratio") is not None else "Historical estimate",
    )
    normalized.setdefault("estimated_image_ratio", float(normalized.get("image_ratio_used", 0.0)))
    normalized.setdefault("base_rf_score", float(normalized.get("hybrid_score", 0.0)))
    normalized.setdefault("ratio_rf_score", float(normalized.get("hybrid_score", 0.0)))
    normalized.setdefault("linear_score", float(normalized.get("hybrid_score", 0.0)))
    normalized.setdefault(
        "fusion_decision",
        {"reason": "Showing a recovered prediction from an earlier session. Run prediction once to refresh it."},
    )
    return normalized


def render_past_data_page(dataframe) -> None:
    st.title("Past Disaster Data")
    st.caption(
        "Historical EM-DAT events for Asia only, limited to floods and storms from January 1, 2000 through December 31, 2025."
    )

    filter_columns = st.columns(3)
    disaster_type_label = filter_columns[0].selectbox("Disaster type", ["All", "Flood", "Storm"], index=0)
    year_range = filter_columns[1].slider("Year range", min_value=2000, max_value=2025, value=(2000, 2025))
    country = filter_columns[2].selectbox("Country", ["All"] + sorted(dataframe["country"].unique().tolist()))

    selected_types = ["Flood", "Storm"] if disaster_type_label == "All" else [disaster_type_label]
    filtered = filter_historical_data(dataframe, disaster_types=selected_types, year_range=year_range, country=country)

    if filtered.empty:
        st.warning("No events match the selected filters.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Events", f"{len(filtered):,}")
    metric_columns[1].metric("Countries", f"{filtered['country'].nunique():,}")
    metric_columns[2].metric("People affected", format_compact_number(filtered["total_affected"].sum()))
    metric_columns[3].metric("Average severity", f"{filtered['severity_score'].mean():.1f}/10")

    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(build_yearly_events_chart(filtered), use_container_width=True)
    chart_columns[1].plotly_chart(build_severity_distribution_chart(filtered), use_container_width=True)

    st.plotly_chart(build_country_summary_chart(filtered), use_container_width=True)

    preview = (
        filtered[
            [
                "start_year",
                "country",
                "subregion",
                "disaster_type",
                "disaster_subtype",
                "event_name",
                "severity_score",
                "total_deaths",
                "total_affected",
                "damage_musd",
            ]
        ]
        .rename(
            columns={
                "start_year": "Year",
                "country": "Country",
                "subregion": "Subregion",
                "disaster_type": "Type",
                "disaster_subtype": "Subtype",
                "event_name": "Event",
                "severity_score": "Severity Score",
                "total_deaths": "Deaths",
                "total_affected": "Affected",
                "damage_musd": "Damage (M US$)",
            }
        )
        .sort_values(["Severity Score", "Affected"], ascending=[False, False])
        .head(25)
    )
    st.subheader("Filtered event table")
    st.dataframe(preview, use_container_width=True, hide_index=True)


def render_map_page(dataframe) -> None:
    st.title("Asia Disaster Map")
    st.caption("An Asia-only map to inspect where flood and storm events cluster across subregions.")

    filter_columns = st.columns(3)
    disaster_types = filter_columns[0].multiselect("Disaster types", ["Flood", "Storm"], default=["Flood", "Storm"])
    year_range = filter_columns[1].slider(
        "Year range",
        min_value=2000,
        max_value=2025,
        value=(2000, 2025),
        key="map_years",
    )
    all_subregions = sorted(dataframe["subregion"].dropna().unique().tolist())
    selected_subregions = filter_columns[2].multiselect("Subregions", all_subregions, default=all_subregions)

    filtered = filter_historical_data(
        dataframe,
        disaster_types=disaster_types or ["Flood", "Storm"],
        year_range=year_range,
        subregions=selected_subregions,
    )

    if filtered.empty:
        st.warning("No mapped events match the selected filters.")
        return

    st.plotly_chart(build_map_figure(filtered), use_container_width=True)

    summary_columns = st.columns(2)
    country_summary = (
        filtered.groupby("country")
        .agg(events=("disno", "size"), avg_severity=("severity_score", "mean"))
        .sort_values(["events", "avg_severity"], ascending=[False, False])
        .head(12)
        .reset_index()
    )
    subregion_summary = (
        filtered.groupby("subregion")
        .agg(events=("disno", "size"), avg_severity=("severity_score", "mean"))
        .sort_values(["events", "avg_severity"], ascending=[False, False])
        .reset_index()
    )

    import plotly.express as px

    country_fig = px.bar(
        country_summary,
        x="country",
        y="events",
        color="events",
        color_continuous_scale=EVENT_COUNT_COLOR_SCALE,
        hover_data={"avg_severity": ":.1f", "events": ":,"},
        labels={"country": "Country", "events": "Events", "avg_severity": "Avg severity (/10)"},
        title="Most active countries",
    )
    country_fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    summary_columns[0].plotly_chart(
        apply_chart_theme(country_fig, show_x_grid=False, show_y_grid=True),
        use_container_width=True,
    )

    subregion_fig = px.bar(
        subregion_summary,
        x="subregion",
        y="events",
        color="events",
        color_continuous_scale=EVENT_COUNT_COLOR_SCALE,
        hover_data={"avg_severity": ":.1f", "events": ":,"},
        labels={"subregion": "Subregion", "events": "Events", "avg_severity": "Avg severity (/10)"},
        title="Subregion activity",
    )
    subregion_fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    summary_columns[1].plotly_chart(
        apply_chart_theme(subregion_fig, show_x_grid=False, show_y_grid=True),
        use_container_width=True,
    )


def get_severity_level(score: float) -> str:
    if score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"

def get_severity_color(score: float) -> str:
    if score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"


def render_prediction_page(predictor: MLRSeverityPredictor, rf_predictor: DisasterSeverityPredictor, legacy_model_summary: Dict[str, str]) -> None:
    st.title("Severity Prediction and Satellite Analysis")
    st.caption(
        "The model is trained on Asia flood and storm events from January 1, 2000 through December 31, 2025. Years above 2025 are forward-looking scenario inputs."
    )

    mode = st.selectbox(
        "Prediction mode",
        [
            "Random Forest (full features)",
            "Random Forest + Image Fusion",
        ],
        index=0,
    )

    st.info("Random Forest uses a richer feature set with 5-Fold Cross-Validation for robust predictions.")
    st.dataframe(rf_predictor.model_quality_table(), use_container_width=True, hide_index=True)

    default_country = "India" if "India" in predictor.available_countries else predictor.available_countries[0]
    disaster_type = st.selectbox("Disaster type", ["Flood", "Storm"], key="pred_type")
    country = st.selectbox(
        "Country",
        predictor.available_countries,
        index=predictor.available_countries.index(default_country),
    )

    _reference_year = 2026
    _start_month = 7
    _disaster_subtype = "N/A"
    _magnitude = 0.0

    scenario_columns = st.columns(4)
    duration_days = scenario_columns[0].number_input("Duration (days)", min_value=1, max_value=180, value=10, step=1)
    total_affected = scenario_columns[1].number_input(
        "Estimated affected population",
        min_value=0,
        max_value=100_000_000,
        value=50_000,
        step=1_000,
    )
    scenario_columns[2].write("")
    scenario_columns[3].write("")

    uploaded_file = st.file_uploader(
        "Upload a flood or storm satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Flood images are checked for water extent. Storm images are checked for building-damage signatures.",
    )

    if st.button("Run prediction", type="primary"):
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
        with st.spinner("Scoring the scenario and analyzing the satellite image..."):
            image_result = None
            if uploaded_bytes:
                image_result = rf_predictor.analyze_satellite_image(uploaded_bytes, disaster_type)

            if mode == "Random Forest + Image Fusion":
                result = rf_predictor.predict(
                    reference_year=_reference_year,
                    disaster_type=disaster_type,
                    country=country,
                    disaster_subtype=_disaster_subtype,
                    start_month=_start_month,
                    duration_days=int(duration_days),
                    total_affected=int(total_affected),
                    magnitude=None,
                    image_bytes=uploaded_bytes,
                )
            else:
                result = rf_predictor.predict(
                    reference_year=_reference_year,
                    disaster_type=disaster_type,
                    country=country,
                    disaster_subtype=_disaster_subtype,
                    start_month=_start_month,
                    duration_days=int(duration_days),
                    total_affected=int(total_affected),
                    magnitude=None,
                    image_bytes=None,
                )
                if image_result:
                    result["image_analysis"] = image_result
                    result["fusion_decision"] = {"use_image": False, "reason": "Image overlay shown; fusion disabled in this mode."}

            st.session_state["prediction_result"] = result
            st.session_state["prediction_image"] = uploaded_bytes

    result = st.session_state.get("prediction_result")
    if not result:
        return

    gauge_info = build_severity_gauge(result["hybrid_score"])
    component_scores = result.get("component_scores", {})
    rf_score = component_scores.get("Random Forest", result["hybrid_score"])
    linear_score = component_scores.get("Linear Regression", rf_score)
    confidence_info = build_confidence_badge(result["confidence"], rf_score, linear_score)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Predicted severity", f"{result['hybrid_score']}/100")
    metric_columns[1].metric("Prediction range", f"{result['prediction_range']['lower']} - {result['prediction_range']['upper']}")
    metric_columns[2].metric("Model", "Random Forest")
    
    severity_html = f"""
    <div style="
        background-color: {gauge_info['color']};
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    ">
    {gauge_info['emoji']} {gauge_info['level']}
    </div>
    """
    metric_columns[3].markdown(severity_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Severity Gauge")
    gauge_col1, gauge_col2 = st.columns([2, 1])
    
    gauge_html = f"""
    <div style="margin-top: 20px;">
        <div style="
            background: linear-gradient(to right, #28a745, #ffc107, #dc3545);
            height: 30px;
            border-radius: 15px;
            position: relative;
            overflow: visible;
        ">
            <div style="
                position: absolute;
                top: -5px;
                left: calc({gauge_info['gauge_value'] * 100}% - 10px);
                width: 0;
                height: 0;
                border-left: 12px solid transparent;
                border-right: 12px solid transparent;
                border-top: 18px solid #333;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px;">
            <span>Low (0)</span>
            <span style="font-weight: bold; color: {gauge_info['color']};">{gauge_info['score']:.1f}</span>
            <span>High (100)</span>
        </div>
    </div>
    """
    gauge_col1.markdown(gauge_html, unsafe_allow_html=True)
    
    confidence_html = f"""
    <div style="
        background-color: {confidence_info['color']};
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    ">
        <div style="font-size: 24px; font-weight: bold;">{confidence_info['confidence']:.0f}%</div>
        <div style="font-size: 12px;">Model Confidence</div>
        <div style="font-size: 11px; margin-top: 5px;">{confidence_info['description']}</div>
    </div>
    """
    gauge_col2.markdown(confidence_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(result["fusion_decision"]["reason"])

    analysis_tabs = st.tabs(["Component Scores", "Feature Importance", "Drivers"])
    
    with analysis_tabs[0]:
        if result.get("component_scores") and len(result["component_scores"]) > 1:
            st.plotly_chart(build_component_chart(result), use_container_width=True)
        else:
            st.info("Component scores only available with multi-model predictions.")
    
    with analysis_tabs[1]:
        st.plotly_chart(build_feature_importance_chart(rf_predictor.feature_importance), use_container_width=True)
        st.caption("Feature importance shows which factors contributed most to the Random Forest prediction.")
    
    with analysis_tabs[2]:
        st.markdown("\n".join(f"- {driver}" for driver in result["drivers"]))

    if result.get("image_analysis"):
        st.markdown("---")
        st.subheader("Satellite Image Analysis")
        
        image_damage_ratio = result.get("image_damage_ratio", 0)
        
        st.markdown("""
        <style>
        .curtain-container {
            position: relative;
            overflow: hidden;
            max-width: 100%;
        }
        .curtain-slider {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 4px;
            background: white;
            cursor: ew-resize;
            z-index: 10;
        }
        .curtain-slider::after {
            content: "⟷";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 5px 8px;
            border-radius: 50%;
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col_before, col_after = st.columns(2)
        
        uploaded_bytes = st.session_state.get("prediction_image")
        if uploaded_bytes:
            col_before.image(uploaded_bytes, caption="Original Image", use_container_width=True)
        
        col_after.image(
            result["image_analysis"]["detection_overlay"],
            caption=f"Detected Damage: {result['image_analysis']['image_damage_score']}/100",
            use_container_width=True,
        )
        
        damage_ratio_color = "#dc3545" if image_damage_ratio > 20 else "#ffc107" if image_damage_ratio > 10 else "#28a745"
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="flex: 1; padding: 15px; background: {damage_ratio_color}; color: white; border-radius: 10px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold;">{image_damage_ratio:.1f}%</div>
                <div style="font-size: 12px;">Damage Coverage Ratio</div>
            </div>
            <div style="flex: 1; padding: 15px; background: #f8f9fa; border-radius: 10px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: #333;">{result['image_analysis']['image_damage_score']:.1f}</div>
                <div style="font-size: 12px; color: #666;">Damage Severity Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_page_v2(
    predictor: MLRSeverityPredictor,
    csv_path: str,
    legacy_model_summary: Dict[str, str],
) -> None:
    st.title("Severity Prediction and Satellite Analysis")
    st.caption(
        "The Random Forest is trained on Asia flood and storm events from 2000–2020 "
        "and evaluated on a held-out temporal test set of 2021–2025 events. "
        "Scenario inputs with years above 2025 are treated as forward-looking."
    )

   
    default_country = "India" if "India" in predictor.available_countries else predictor.available_countries[0]
    disaster_type = st.selectbox("Disaster type", ["Flood", "Storm"], key="pred_type_v2")
    country = st.selectbox(
        "Country",
        predictor.available_countries,
        index=predictor.available_countries.index(default_country),
        key="pred_country_v2",
    )

    reference_year = 2026
    start_month = 7
    disaster_subtype = "N/A"

    scenario_columns = st.columns(4)
    duration_days = scenario_columns[0].number_input(
        "Duration (days)",
        min_value=1,
        max_value=180,
        value=10,
        step=1,
        key="duration_days_v2",
    )
    total_affected = scenario_columns[1].number_input(
        "Estimated affected population",
        min_value=0,
        max_value=100_000_000,
        value=50_000,
        step=1_000,
        key="total_affected_v2",
    )
    scenario_columns[2].metric("Severity scale", "0 to 10")
    scenario_columns[3].metric("Image ratio input", "Auto")

    uploaded_file = st.file_uploader(
        "Upload a flood or storm satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="If no image is uploaded, the model uses a historical ratio estimate from past disaster data.",
        key="pred_upload_v2",
    )

    if st.button("Run prediction", type="primary", key="run_prediction_v2"):
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
        spinner_text = (
            "Loading cached prediction model and scoring the scenario..."
            if has_cached_rf_predictor(csv_path)
            else "Building prediction model for the first run and scoring the scenario..."
        )
        with st.spinner(spinner_text):
            rf_predictor = get_rf_predictor(csv_path, CACHE_VERSION)
            result = rf_predictor.predict(
                reference_year=reference_year,
                disaster_type=disaster_type,
                country=country,
                disaster_subtype=disaster_subtype,
                start_month=start_month,
                duration_days=int(duration_days),
                total_affected=int(total_affected),
                magnitude=None,
                image_bytes=uploaded_bytes,
            )
            st.session_state["prediction_result_v2"] = result
            st.session_state["prediction_image_v2"] = uploaded_bytes

    result = normalize_prediction_result(st.session_state.get("prediction_result_v2"))
    if not result:
        return
    st.session_state["prediction_result_v2"] = result
    rf_predictor = get_rf_predictor(csv_path, CACHE_VERSION)

    gauge_info = build_severity_gauge(result["hybrid_score"])
    confidence_info = build_confidence_badge(
        result["confidence"],
        result.get("ratio_rf_score", result["hybrid_score"]),
        result.get("linear_score", result["hybrid_score"]),
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Predicted severity", f"{result['hybrid_score']:.1f}/10")
    metric_columns[1].metric(
        "Prediction range",
        f"{result['prediction_range']['lower']:.1f} - {result['prediction_range']['upper']:.1f}",
    )
    metric_columns[2].metric("Ratio used in model", f"{result['image_ratio_used']:.1f}%")
    metric_columns[3].metric("Ratio source", result["image_ratio_source"])

    severity_html = f"""
    <div style="
        background: linear-gradient(90deg, {APP_PALETTE["accent"]} 0%, {gauge_info["color"]} 100%);
        color: #13232f;
        padding: 12px 18px;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        margin-top: 8px;
    ">
        Final Severity: {gauge_info["score"]:.1f}/10 ({gauge_info["level"]})
    </div>
    """
    st.markdown(severity_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Severity Gauge")
    gauge_col1, gauge_col2 = st.columns([2, 1])
    gauge_html = f"""
    <div style="margin-top: 20px;">
        <div style="
            background: linear-gradient(to right, {APP_PALETTE["low"]}, {APP_PALETTE["medium"]}, {APP_PALETTE["high"]});
            height: 30px;
            border-radius: 15px;
            position: relative;
            overflow: visible;
        ">
            <div style="
                position: absolute;
                top: -5px;
                left: calc({gauge_info['gauge_value'] * 100}% - 10px);
                width: 0;
                height: 0;
                border-left: 12px solid transparent;
                border-right: 12px solid transparent;
                border-top: 18px solid #264653;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px;">
            <span>Low (0)</span>
            <span style="font-weight: bold; color: {gauge_info['color']};">{gauge_info['score']:.1f}/10</span>
            <span>High (10)</span>
        </div>
    </div>
    """
    gauge_col1.markdown(gauge_html, unsafe_allow_html=True)

    confidence_html = f"""
    <div style="
        background-color: {confidence_info['color']};
        color: #13232f;
        padding: 15px;
        border-radius: 14px;
        text-align: center;
        margin-top: 10px;
    ">
        <div style="font-size: 24px; font-weight: bold;">{confidence_info['confidence']:.0f}%</div>
        <div style="font-size: 12px;">Model Confidence</div>
        <div style="font-size: 11px; margin-top: 5px;">{confidence_info['description']}</div>
    </div>
    """
    gauge_col2.markdown(confidence_html, unsafe_allow_html=True)

    ratio_columns = st.columns(3)
    ratio_columns[0].metric("Historical ratio estimate", f"{result['estimated_image_ratio']:.1f}%")
    ratio_columns[1].metric("Ratio-aware RF score", f"{result['ratio_rf_score']:.1f}/10")
    ratio_columns[2].metric("Past-data RF score", f"{result['base_rf_score']:.1f}/10")

    with st.expander("Model Quality & Validation", expanded=False):
        holdout = getattr(rf_predictor, "holdout_info", {})
        if holdout:
            strategy = holdout.get("strategy", "")
            if strategy == "temporal":
                st.caption(
                    f"Temporal holdout — trained on {holdout['train_years']} "
                    f"({holdout['train_size']:,} records), tested on {holdout['test_years']} "
                    f"({holdout['test_size']:,} records). "
                    "This is a stricter evaluation than a random split because the model "
                    "never sees post-2020 events during training."
                )
            else:
                st.caption(
                    f"Random 80/20 holdout used (temporal test set had fewer than 30 records). "
                    f"Train size: {holdout.get('train_size', '?')}, "
                    f"Test size: {holdout.get('test_size', '?')}."
                )
        st.dataframe(rf_predictor.model_quality_table(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(result["fusion_decision"]["reason"])

    analysis_tabs = st.tabs(["Component Scores", "Feature Importance", "Drivers", "Historical Matches", "Image Detector Calibration"])

    with analysis_tabs[0]:
        if result.get("component_scores") and len(result["component_scores"]) > 1:
            st.plotly_chart(build_component_chart(result), use_container_width=True)
        else:
            st.info("Component scores only available with multi-model predictions.")

    with analysis_tabs[1]:
        st.plotly_chart(build_feature_importance_chart(rf_predictor.feature_importance), use_container_width=True)
        st.caption(
            "Feature importance now includes the image-ratio input used by the Random Forest."
        )

    with analysis_tabs[2]:
        st.markdown("\n".join(f"- {driver}" for driver in result["drivers"]))

    with analysis_tabs[3]:
        similar_events = result.get("similar_events")
        if similar_events is not None and not similar_events.empty:
            st.dataframe(similar_events, use_container_width=True, hide_index=True)
        else:
            st.info("No close historical matches were found for this scenario.")

    with analysis_tabs[4]:
        st.caption(
            "Runs the flood and storm image detectors on 7 synthetic reference images "
            "with known expected outcomes. A 'YES' pass means the detector correctly "
            "identified the coverage level. This validates that the heuristic CV pipeline "
            "is not systematically wrong before trusting its output in predictions."
        )
        calib_df = rf_predictor.calibrate_image_analysis()
        # Colour the Pass column for quick scanning
        def _highlight_pass(val):
            return "color: green; font-weight: bold" if val == "YES" else "color: red; font-weight: bold"
        st.dataframe(
            calib_df.style.applymap(_highlight_pass, subset=["Pass"]),
            use_container_width=True,
            hide_index=True,
        )
        pass_count = (calib_df["Pass"] == "YES").sum()
        total = len(calib_df)
        st.metric("Calibration pass rate", f"{pass_count}/{total}", f"{pass_count/total*100:.0f}%")

    if result.get("image_analysis"):
        st.markdown("---")
        st.subheader("Satellite Image Analysis")
        image_damage_ratio = float(result.get("image_damage_ratio") or 0.0)
        col_before, col_after = st.columns(2)
        uploaded_bytes = st.session_state.get("prediction_image_v2")
        if uploaded_bytes:
            col_before.image(uploaded_bytes, caption="Original Image", use_container_width=True)

        col_after.image(
            result["image_analysis"]["detection_overlay"],
            caption=f"Detected image severity: {result['image_analysis']['image_damage_score']:.1f}/10",
            use_container_width=True,
        )

        damage_ratio_color = (
            APP_PALETTE["high"]
            if image_damage_ratio > 20
            else APP_PALETTE["medium"]
            if image_damage_ratio > 10
            else APP_PALETTE["low"]
        )
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div style="flex: 1; padding: 15px; background: {damage_ratio_color}; color: white; border-radius: 12px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold;">{image_damage_ratio:.1f}%</div>
                    <div style="font-size: 12px;">Image Processing Ratio</div>
                </div>
                <div style="flex: 1; padding: 15px; background: {APP_PALETTE["surface"]}; border: 1px solid {APP_PALETTE["border"]}; border-radius: 12px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: {APP_PALETTE["text"]};">{result['image_analysis']['image_damage_score']:.1f}</div>
                    <div style="font-size: 12px; color: {APP_PALETTE["muted"]};">Image-only severity (/10)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Asia Disaster Intelligence", page_icon=":earth_asia:", layout="wide")
    inject_theme_styles()

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    historical_df = get_historical_data(str(DATA_PATH), CACHE_VERSION)
    predictor = get_predictor(str(DATA_PATH), CACHE_VERSION)
    legacy_model_summary = get_legacy_summary(str(LEGACY_MODEL_PATH), CACHE_VERSION)

    if float(historical_df["severity_score"].mean()) > SEVERITY_MAX:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_result_v2", None)
        st.rerun()

    st.sidebar.title("Asia Disaster Intelligence")
    st.sidebar.caption("Flood and storm analytics for Asia, 2000–2025. Model trained on 2000–2020.")
    page = st.sidebar.radio("Open page", ["Past Data", "Asia Map", "Prediction"])

    st.sidebar.metric("Records", f"{len(historical_df):,}")
    st.sidebar.metric("Countries", f"{historical_df['country'].nunique():,}")
    st.sidebar.metric("Average severity", f"{historical_df['severity_score'].mean():.1f}/10")
    
    if page == "Past Data":
        render_past_data_page(historical_df)
    elif page == "Asia Map":
        render_map_page(historical_df)
    else:
        render_prediction_page_v2(
            predictor,
            str(DATA_PATH),
            legacy_model_summary,
        )


if __name__ == "__main__":
    main()
