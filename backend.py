"""
Backend logic for the Asia Disaster Intelligence Streamlit app.

This module contains:
- data preparation + coordinate resolution
- the severity predictor model (linear + random forest + image analysis fusion)
- plotly chart builders used by the UI
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEVERITY_MAX = 10.0
IMAGE_RATIO_MAX = 100.0
GAUGE_HIGH_THRESHOLD = 7.0
GAUGE_MEDIUM_THRESHOLD = 4.0
CV_SPLITS = 3

ASIA_FALLBACK_COORDINATES = {
    "Armenia": (40.07, 45.04),
    "Azerbaijan": (40.14, 47.58),
    "Bhutan": (27.51, 90.43),
    "China, Hong Kong Special Administrative Region": (22.32, 114.17),
    "China, Macao Special Administrative Region": (22.20, 113.55),
    "Cyprus": (35.12, 33.36),
    "Jordan": (31.95, 35.93),
    "Kuwait": (29.38, 47.98),
    "Lebanon": (33.89, 35.50),
    "Maldives": (3.20, 73.22),
    "Qatar": (25.29, 51.53),
    "Syrian Arab Republic": (33.51, 36.29),
    "Timor-Leste": (-8.56, 125.57),
    "United Arab Emirates": (24.45, 54.37),
}

APP_PALETTE = {
    "background": "#ffffff",
    "surface": "#ffffff",
    "chart_surface": "#ffffff",
    "sidebar_start": "#ffffff",
    "sidebar_end": "#ffffff",
    "text": "#0f172a",
    "muted": "#475569",
    "border": "#d7dee8",
    "grid": "#e2e8f0",
    "accent": "#f4b000",
    "accent_soft": "#fde7a0",
    "flood": "#0f766e",
    "storm": "#c2410c",
    "low": "#166534",
    "medium": "#b45309",
    "high": "#b91c1c",
    "ocean": "#dbeafe",
    "land": "#e5e7eb",
}

DISASTER_COLORS = {"Flood": APP_PALETTE["flood"], "Storm": APP_PALETTE["storm"]}
SEVERITY_COLOR_SCALE = [
    "#fde68a",
    "#f59e0b",
    "#f97316",
    "#dc2626",
    "#7f1d1d",
]
EVENT_COUNT_COLOR_SCALE = [
    "#dbeafe",
    "#93c5fd",
    "#3b82f6",
    "#1d4ed8",
    "#0f172a",
]
FEATURE_IMPORTANCE_SCALE = ["#fde68a", "#f59e0b", "#0ea5e9", "#111827"]


def severity_band_label(score: float) -> str:
    if score >= 7.5:
        return "Extreme"
    if score >= 5.5:
        return "High"
    if score >= 3.5:
        return "Moderate"
    return "Low"


def format_compact_number(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values.reshape(-1), level))


def _prepare_scene_features(rgb_image: np.ndarray) -> Dict[str, np.ndarray]:
    rgb_u8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    lab_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    gray_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    red = rgb_u8[:, :, 0].astype(np.float32) / 255.0
    green = rgb_u8[:, :, 1].astype(np.float32) / 255.0
    blue = rgb_u8[:, :, 2].astype(np.float32) / 255.0
    hue = hsv_u8[:, :, 0].astype(np.float32)
    saturation = hsv_u8[:, :, 1].astype(np.float32) / 255.0
    value = hsv_u8[:, :, 2].astype(np.float32) / 255.0
    lab_b = lab_u8[:, :, 2].astype(np.float32) - 128.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (5, 5), 0)

    smoothed_gray = cv2.GaussianBlur(gray, (9, 9), 0)
    smoothed_gray_sq = cv2.GaussianBlur(gray * gray, (9, 9), 0)
    local_std = np.sqrt(np.clip(smoothed_gray_sq - smoothed_gray * smoothed_gray, 0, None))

    edges = cv2.Canny(gray_u8, 60, 140).astype(np.float32) / 255.0
    edge_density = cv2.GaussianBlur(edges, (9, 9), 0)

    sky_mask = (value >= 0.88) & (saturation <= 0.15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 127

    return {
        "rgb_u8": rgb_u8,
        "gray_u8": gray_u8,
        "gray": gray,
        "red": red,
        "green": green,
        "blue": blue,
        "hue": hue,
        "saturation": saturation,
        "value": value,
        "lab_b": lab_b,
        "gradient": gradient,
        "local_std": local_std,
        "edge_density": edge_density,
        "sky_mask": sky_mask,
    }


def _cleanup_mask(
    mask: np.ndarray,
    *,
    open_size: int = 5,
    close_size: int = 11,
    min_area_ratio: float = 0.0006,
) -> np.ndarray:
    cleaned = mask.astype(np.uint8) * 255
    if open_size > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        )
    if close_size > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        )

    min_area = max(80, int(cleaned.size * min_area_ratio))
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    filtered = np.zeros_like(cleaned)
    for label in range(1, label_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered > 0


def _component_touches_border(
    stats: np.ndarray, label: int, image_width: int, image_height: int
) -> bool:
    left = int(stats[label, cv2.CC_STAT_LEFT])
    top = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    return (
        left == 0
        or top == 0
        or left + width >= image_width
        or top + height >= image_height
    )


def _flood_water_analysis(rgb_image: np.ndarray) -> Dict[str, object]:
    scene = _prepare_scene_features(rgb_image)
    hue = scene["hue"]
    saturation = scene["saturation"]
    value = scene["value"]
    red = scene["red"]
    green = scene["green"]
    blue = scene["blue"]
    lab_b = scene["lab_b"]
    gradient = scene["gradient"]
    local_std = scene["local_std"]
    edge_density = scene["edge_density"]
    sky_mask = scene["sky_mask"]

    gradient_low = _quantile(gradient, 0.45)
    gradient_mid = _quantile(gradient, 0.62)
    std_low = _quantile(local_std, 0.50)
    std_mid = _quantile(local_std, 0.65)
    edge_mid = _quantile(edge_density, 0.65)

    smooth = (gradient <= gradient_mid) & (local_std <= std_mid)
    very_smooth = (gradient <= gradient_low) & (local_std <= std_low)

    vegetation = (
        (hue >= 28)
        & (hue <= 96)
        & (saturation >= 0.20)
        & (green >= red * 1.04)
    )
    bright_cloud = (value >= 0.90) & (saturation <= 0.10)

    blue_seed = (hue >= 82) & (hue <= 138) & (saturation >= 0.11) & (value >= 0.12)
    muddy_seed = (
        (hue >= 8)
        & (hue <= 28)
        & (saturation >= 0.08)
        & (saturation <= 0.50)
        & (value >= 0.12)
        & (value <= 0.88)
        & (lab_b >= 2.0)
        & (red >= blue * 1.02)
        & (green >= blue * 0.98)
        & (np.abs(red - green) <= 0.18)
    )
    dark_seed = (value <= 0.20) & (saturation <= 0.20)

    land_background = (value <= 0.85) & (saturation >= 0.08)
    
    seed_mask = (blue_seed | muddy_seed | dark_seed) & smooth & ~vegetation & ~bright_cloud & ~sky_mask & land_background
    seed_mask = _cleanup_mask(seed_mask, open_size=5, close_size=7, min_area_ratio=0.00025)
    seed_u8 = seed_mask.astype(np.uint8) * 255
    expanded_seed = (
        cv2.dilate(
            seed_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
            iterations=1,
        )
        > 0
    )

    blue_candidate = (hue >= 78) & (hue <= 145) & (saturation >= 0.05) & (value >= 0.10)
    muddy_candidate = (
        (hue >= 6)
        & (hue <= 32)
        & (saturation >= 0.05)
        & (saturation <= 0.58)
        & (value >= 0.10)
        & (value <= 0.92)
        & (lab_b >= -1.0)
        & (red >= blue * 0.96)
        & (green >= blue * 0.90)
    )
    gray_candidate = (
        (saturation <= 0.15)
        & (value >= 0.18)
        & (value <= 0.92)
        & (np.abs(red - green) <= 0.10)
        & (np.abs(green - blue) <= 0.12)
    )
    dark_candidate = (value <= 0.22) & (saturation <= 0.24)

    candidate_mask = (
        (((blue_candidate | muddy_candidate | dark_candidate) & smooth)
        | (gray_candidate & very_smooth & expanded_seed))
        & ~vegetation
        & ~bright_cloud
        & ~sky_mask
        & land_background
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8)
    filtered = np.zeros_like(candidate_u8)
    min_area = max(100, int(candidate_u8.size * 0.0007))
    large_area = max(1_000, int(candidate_u8.size * 0.0100))
    image_height, image_width = candidate_u8.shape
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == label
        overlap = int(np.count_nonzero(seed_u8[component]))
        mean_gradient = float(gradient[component].mean())
        mean_std = float(local_std[component].mean())
        mean_edge = float(edge_density[component].mean())
        mean_hue = float(hue[component].mean())
        mean_sat = float(saturation[component].mean())
        mean_lab_b = float(lab_b[component].mean())
        color_like = (
            (78 <= mean_hue <= 145 and mean_sat >= 0.05)
            or (6 <= mean_hue <= 32 and mean_lab_b >= -1.0)
            or mean_sat <= 0.14
        )
        smooth_component = (
            mean_gradient <= gradient_mid
            and mean_std <= std_mid
            and mean_edge <= edge_mid
        )
        if overlap >= max(20, int(area * 0.01)):
            filtered[component] = 255
        elif (
            area >= large_area
            and smooth_component
            and color_like
            and (
                _component_touches_border(stats, label, image_width, image_height)
                or mean_sat <= 0.16
                or (6 <= mean_hue <= 32 and mean_lab_b >= 3.0)
            )
        ):
            filtered[component] = 255

    mask = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    mask = mask > 0

    land_pixels = ~sky_mask & land_background
    land_coverage = float(mask[land_pixels].mean()) if land_pixels.any() else float(mask.mean())
    
    mask_coverage = float(mask.mean())
    seed_coverage = float(seed_mask.mean())
    smooth_surface_ratio = float(np.mean(very_smooth[mask])) if mask.any() else 0.0
    blue_ratio = float(np.mean(blue_candidate[mask])) if mask.any() else 0.0
    gray_ratio = float(np.mean(gray_candidate[mask])) if mask.any() else 0.0
    muddy_ratio = float(np.mean(muddy_candidate[mask])) if mask.any() else 0.0
    open_water_signal = float(np.clip(land_coverage / 0.20, 0.0, 1.0))
    seed_signal = float(np.clip(seed_coverage / 0.04, 0.0, 1.0))
    smooth_signal = float(np.clip(smooth_surface_ratio / 0.85, 0.0, 1.0))
    blue_signal = float(np.clip(blue_ratio / 0.60, 0.0, 1.0))
    gray_signal = float(np.clip(gray_ratio / 0.35, 0.0, 1.0))
    muddy_signal = float(np.clip(muddy_ratio / 0.45, 0.0, 1.0))
    tone_signal = max(blue_signal, gray_signal, muddy_signal)

    image_score = SEVERITY_MAX * (
        0.52 * open_water_signal
        + 0.20 * tone_signal
        + 0.16 * smooth_signal
        + 0.12 * seed_signal
    )
    analysis_confidence = 100.0 * (
        0.40 * open_water_signal
        + 0.25 * seed_signal
        + 0.20 * smooth_signal
        + 0.15 * tone_signal
    )

    notes: List[str] = []
    sky_percentage = float(sky_mask.mean()) * 100
    if sky_percentage > 30:
        notes.append(f"significant sky region ({sky_percentage:.0f}%) excluded from analysis")
    if open_water_signal >= 0.60:
        notes.append("broad flood-water coverage is visible")
    elif open_water_signal >= 0.28:
        notes.append("localized flood-water pockets are visible")
    if muddy_signal >= max(blue_signal, gray_signal) and muddy_signal >= 0.40:
        notes.append("muddy sediment-heavy water is dominant")
    elif gray_signal >= max(blue_signal, muddy_signal) and gray_signal >= 0.35:
        notes.append("gray floodwater and washout signatures are present")
    elif blue_signal >= 0.35:
        notes.append("clearer blue or blue-green water surfaces are present")
    if smooth_signal >= 0.55:
        notes.append("the detected water areas are smoother than nearby land")
    if not notes:
        notes.append("water cues are present but remain limited in extent")

    return {
        "mask": mask,
        "image_damage_score": float(np.clip(image_score, 0.0, SEVERITY_MAX)),
        "analysis_confidence": round(float(np.clip(analysis_confidence, 0.0, 100.0)), 1),
        "summary": "Image-only flood analysis suggests that " + ", ".join(notes) + ".",
        "signals": {
            "Water extent (land-only)": round(open_water_signal * 100, 1),
            "Muddy water": round(muddy_signal * 100, 1),
            "Gray floodwater": round(gray_signal * 100, 1),
            "Blue-water likelihood": round(blue_signal * 100, 1),
            "Surface smoothness": round(smooth_signal * 100, 1),
        },
        "overlay_caption": "Detected flood-water footprint (sky excluded)",
        "overlay_color": (35, 181, 211),
        "coverage_label": "Water coverage (land)",
        "score_label": "Flood damage score (image only, /10)",
    }


def _storm_damage_analysis(rgb_image: np.ndarray) -> Dict[str, object]:
    scene = _prepare_scene_features(rgb_image)
    hue = scene["hue"]
    saturation = scene["saturation"]
    value = scene["value"]
    red = scene["red"]
    green = scene["green"]
    blue = scene["blue"]
    gradient = scene["gradient"]
    local_std = scene["local_std"]
    edge_density = scene["edge_density"]
    gray = scene["gray"]

    gradient_mid = _quantile(gradient, 0.62)
    gradient_high = _quantile(gradient, 0.78)
    std_mid = _quantile(local_std, 0.58)
    std_high = _quantile(local_std, 0.76)
    edge_mid = _quantile(edge_density, 0.68)
    edge_high = _quantile(edge_density, 0.84)

    vegetation = (
        (hue >= 28)
        & (hue <= 96)
        & (saturation >= 0.20)
        & (green >= red * 1.04)
    )
    bright_cloud = (value >= 0.92) & (saturation <= 0.12)

    debris_mask = (
        (
            (saturation <= 0.24)
            & (value >= 0.22)
            & (value <= 0.88)
            & (np.abs(red - green) <= 0.16)
            & (np.abs(green - blue) <= 0.18)
        )
        | (
            (hue >= 8)
            & (hue <= 28)
            & (saturation >= 0.10)
            & (saturation <= 0.55)
            & (value >= 0.18)
            & (value <= 0.82)
        )
    ) & ~vegetation

    exposed_ground_mask = (
        (hue >= 7)
        & (hue <= 26)
        & (saturation >= 0.12)
        & (saturation <= 0.62)
        & (value >= 0.16)
        & (value <= 0.84)
        & (red >= blue * 1.02)
    ) & ~vegetation
    bright_rubble_mask = (
        (saturation <= 0.18)
        & (value >= 0.48)
        & (local_std >= std_mid)
        & (local_std <= std_high * 1.15)
    ) & ~vegetation
    shadow_damage_mask = (
        (value <= 0.24) & (edge_density >= edge_mid) & (gradient >= gradient_mid)
    ) & ~vegetation

    fragmentation_seed = ((edge_density >= edge_high) & (gradient >= gradient_high)) & ~vegetation & ~bright_cloud
    damage_seed = fragmentation_seed & (
        debris_mask | exposed_ground_mask | bright_rubble_mask | shadow_damage_mask
    )
    damage_seed = _cleanup_mask(damage_seed, open_size=3, close_size=5, min_area_ratio=0.00018)
    expanded_seed = (
        cv2.dilate(
            damage_seed.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        > 0
    )

    candidate_mask = (
        (
            (debris_mask | exposed_ground_mask | bright_rubble_mask | shadow_damage_mask)
            & ((edge_density >= edge_mid) | expanded_seed)
            & (gradient >= gradient_mid * 0.75)
        )
        & ~bright_cloud
        & ~vegetation
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8)
    filtered = np.zeros_like(candidate_u8)
    min_area = max(50, int(candidate_u8.size * 0.0002))
    large_area = max(400, int(candidate_u8.size * 0.0030))
    image_height, image_width = candidate_u8.shape
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == label
        overlap = int(np.count_nonzero(damage_seed[component]))
        mean_gradient = float(gradient[component].mean())
        mean_std = float(local_std[component].mean())
        mean_edge = float(edge_density[component].mean())
        mean_sat = float(saturation[component].mean())
        mean_value = float(value[component].mean())
        component_is_damage_like = (
            mean_edge >= edge_mid
            and mean_gradient >= gradient_mid * 0.75
            and mean_std >= std_mid * 0.85
            and mean_value >= 0.16
            and mean_value <= 0.90
            and mean_sat <= 0.42
        )
        if overlap >= max(12, int(area * 0.015)):
            filtered[component] = 255
        elif (
            area >= large_area
            and component_is_damage_like
            and not _component_touches_border(stats, label, image_width, image_height)
        ):
            filtered[component] = 255

    mask = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    mask = mask > 0

    mask_coverage = float(mask.mean())
    debris_ratio = float(np.mean(debris_mask[mask])) if mask.any() else 0.0
    exposed_ratio = float(np.mean(exposed_ground_mask[mask])) if mask.any() else 0.0
    rubble_ratio = float(np.mean(bright_rubble_mask[mask])) if mask.any() else 0.0
    fragmentation_ratio = float(np.mean(fragmentation_seed[mask])) if mask.any() else 0.0
    shadow_ratio = float(np.mean(shadow_damage_mask[mask])) if mask.any() else 0.0
    contrast_signal = float(np.clip(float(gray.std()) / 0.22, 0.0, 1.0))
    damage_extent_signal = float(np.clip(mask_coverage / 0.10, 0.0, 1.0))
    debris_signal = float(np.clip(debris_ratio / 0.55, 0.0, 1.0))
    exposed_signal = float(np.clip(exposed_ratio / 0.45, 0.0, 1.0))
    rubble_signal = float(np.clip(rubble_ratio / 0.35, 0.0, 1.0))
    fragmentation_signal = float(np.clip(fragmentation_ratio / 0.18, 0.0, 1.0))
    shadow_signal = float(np.clip(shadow_ratio / 0.18, 0.0, 1.0))

    building_damage_signal = float(
        np.clip(
            0.32 * damage_extent_signal
            + 0.24 * fragmentation_signal
            + 0.20 * debris_signal
            + 0.14 * rubble_signal
            + 0.10 * exposed_signal,
            0.0,
            1.0,
        )
    )
    image_score = SEVERITY_MAX * building_damage_signal
    analysis_confidence = 100.0 * (
        0.35 * fragmentation_signal
        + 0.22 * debris_signal
        + 0.18 * damage_extent_signal
        + 0.12 * rubble_signal
        + 0.08 * shadow_signal
        + 0.05 * contrast_signal
    )

    notes: List[str] = []
    if damage_extent_signal >= 0.55:
        notes.append("damage footprints cover a broad portion of the scene")
    elif damage_extent_signal >= 0.25:
        notes.append("damage footprints appear in localized clusters")
    if fragmentation_signal >= 0.45:
        notes.append("structural fragmentation is elevated")
    if debris_signal >= 0.40:
        notes.append("debris-colored surfaces are elevated")
    if rubble_signal >= 0.35:
        notes.append("bright rubble and roof-scatter cues are present")
    if shadow_signal >= 0.28:
        notes.append("collapsed-shadow cues appear near damaged zones")
    if not notes:
        notes.append("damage cues are present but remain visually weak")

    return {
        "mask": mask,
        "image_damage_score": float(np.clip(image_score, 0.0, SEVERITY_MAX)),
        "analysis_confidence": round(float(np.clip(analysis_confidence, 0.0, 100.0)), 1),
        "summary": "Image-only storm analysis suggests that " + ", ".join(notes) + ".",
        "signals": {
            "Damage extent": round(damage_extent_signal * 100, 1),
            "Structural fragmentation": round(fragmentation_signal * 100, 1),
            "Debris texture": round(debris_signal * 100, 1),
            "Exposed ground": round(exposed_signal * 100, 1),
            "Collapsed-shadow cues": round(shadow_signal * 100, 1),
        },
        "overlay_caption": "Detected building-damage footprint",
        "overlay_color": (239, 96, 56),
        "coverage_label": "Damage coverage",
        "score_label": "Storm damage score (image only, /10)",
    }


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Pre-monsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _scaled_log_signal(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    transformed = np.log1p(series)
    positive = transformed[transformed > 0]
    upper = float(positive.quantile(0.99)) if not positive.empty else 1.0
    upper = max(upper, 1.0)
    return np.clip(transformed / upper, 0, 1)


def _build_historical_image_ratio(
    dataframe: pd.DataFrame,
    deaths_signal: pd.Series,
    affected_signal: pd.Series,
    damage_signal: pd.Series,
) -> pd.Series:
    duration_signal = _scaled_log_signal(dataframe["duration_days"])
    flood_boost = np.where(dataframe["disaster_type"].eq("Flood"), 1.08, 0.96)
    monsoon_boost = np.where(dataframe["season"].eq("Monsoon"), 1.05, 1.0)
    ratio = IMAGE_RATIO_MAX * (
        0.42 * affected_signal
        + 0.26 * damage_signal
        + 0.18 * deaths_signal
        + 0.14 * duration_signal
    )
    return pd.Series(
        np.clip(ratio * flood_boost * monsoon_boost, 0.0, IMAGE_RATIO_MAX),
        index=dataframe.index,
    )


def _build_duration_days(dataframe: pd.DataFrame) -> pd.Series:
    start_year = dataframe["start_year"].fillna(2000).astype(int)
    start_month = dataframe["start_month"].fillna(1).clip(1, 12).astype(int)
    start_day = dataframe["start_day"].fillna(1).clip(1, 28).astype(int)

    end_year = dataframe["end_year"].fillna(dataframe["start_year"]).fillna(2000)
    end_year = end_year.astype(int)
    end_month = dataframe["end_month"].fillna(dataframe["start_month"]).fillna(1)
    end_month = end_month.clip(1, 12).astype(int)
    end_day = dataframe["end_day"].fillna(dataframe["start_day"]).fillna(1)
    end_day = end_day.clip(1, 28).astype(int)

    start_dates = pd.to_datetime(
        {"year": start_year, "month": start_month, "day": start_day},
        errors="coerce",
    )
    end_dates = pd.to_datetime(
        {"year": end_year, "month": end_month, "day": end_day},
        errors="coerce",
    )

    fallback = ((end_year - start_year).clip(lower=0) * 365 + 7).astype(float)
    duration = (end_dates - start_dates).dt.days.add(1).fillna(fallback)
    return duration.clip(lower=1, upper=180)


def _resolve_map_coordinates(dataframe: pd.DataFrame) -> pd.DataFrame:
    resolved = dataframe.copy()
    country_medians = (
        resolved.dropna(subset=["latitude", "longitude"])
        .groupby("country")[["latitude", "longitude"]]
        .median()
    )

    resolved["map_latitude"] = resolved["latitude"]
    resolved["map_longitude"] = resolved["longitude"]

    missing = resolved["map_latitude"].isna() | resolved["map_longitude"].isna()
    for index in resolved.index[missing]:
        country = resolved.at[index, "country"]
        if country in country_medians.index:
            resolved.at[index, "map_latitude"] = country_medians.at[country, "latitude"]
            resolved.at[index, "map_longitude"] = country_medians.at[country, "longitude"]
        elif country in ASIA_FALLBACK_COORDINATES:
            latitude, longitude = ASIA_FALLBACK_COORDINATES[country]
            resolved.at[index, "map_latitude"] = latitude
            resolved.at[index, "map_longitude"] = longitude

    seed = pd.util.hash_pandas_object(resolved["disno"].astype(str), index=False).astype("uint64")
    lat_jitter = ((seed % 11).astype(float) - 5.0) * 0.18
    lon_jitter = (((seed // 11) % 11).astype(float) - 5.0) * 0.22

    has_map = resolved["map_latitude"].notna() & resolved["map_longitude"].notna()
    resolved.loc[has_map, "map_latitude"] = resolved.loc[has_map, "map_latitude"] + lat_jitter[has_map]
    resolved.loc[has_map, "map_longitude"] = resolved.loc[has_map, "map_longitude"] + lon_jitter[has_map]
    return resolved


def _build_detection_overlay(
    rgb_image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]
) -> Image.Image:
    base = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    highlighted = base.copy()
    if mask.any():
        color_layer = np.zeros_like(base)
        color_layer[:, :] = np.array(color, dtype=np.uint8)
        highlighted[mask] = (0.35 * base[mask] + 0.65 * color_layer[mask]).astype(np.uint8)
    return Image.fromarray(highlighted)


def _filter_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    
    zero_duration_high_affected = (
        (df["duration_days"] == 0) & 
        (df["total_affected"] > 1000)
    )
    
    extreme_deaths_low_affected = (
        (df["total_deaths"] > 1000) & 
        (df["total_affected"] < 10)
    )
    
    zero_impact = (
        (df["total_deaths"] == 0) & 
        (df["total_affected"] == 0) & 
        (df["damage_k"] == 0)
    )
    
    impossible_duration = (
        (df["duration_days"] < 0) | 
        (df["duration_days"] > 365)
    )
    
    invalid_year = (
        (df["start_year"] < 2000) | 
        (df["start_year"] > 2025)
    )
    
    mask = ~(zero_duration_high_affected | extreme_deaths_low_affected | 
             zero_impact | impossible_duration | invalid_year)
    
    removed_count = (~mask).sum()
    if removed_count > 0:
        print(f"Filtered {removed_count} outlier records from training data")
    
    return df[mask].reset_index(drop=True)


def load_and_prepare_disaster_data(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    disno_fallback = pd.Series(raw.index.astype(str), index=raw.index)

    prepared = pd.DataFrame(
        {
            "disno": raw["DisNo."].fillna(disno_fallback).astype(str),
            "disaster_type": raw["Disaster Type"].fillna("Unknown").astype(str),
            "disaster_subtype": raw["Disaster Subtype"].fillna("Unspecified").astype(str),
            "country": raw["Country"].fillna("Unknown").astype(str),
            "subregion": raw["Subregion"].fillna("Unknown").astype(str),
            "region": raw["Region"].fillna("Unknown").astype(str),
            "event_name": raw["Event Name"].fillna("").astype(str).str.strip(),
            "location": raw["Location"].fillna("Location not reported").astype(str),
            "start_year": pd.to_numeric(raw["Start Year"], errors="coerce"),
            "start_month": pd.to_numeric(raw["Start Month"], errors="coerce").fillna(1),
            "start_day": pd.to_numeric(raw["Start Day"], errors="coerce").fillna(1),
            "end_year": pd.to_numeric(raw["End Year"], errors="coerce"),
            "end_month": pd.to_numeric(raw["End Month"], errors="coerce"),
            "end_day": pd.to_numeric(raw["End Day"], errors="coerce"),
            "magnitude": pd.to_numeric(raw["Magnitude"], errors="coerce"),
            "latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "total_deaths": pd.to_numeric(raw["Total Deaths"], errors="coerce").fillna(0),
            "total_affected": pd.to_numeric(raw["Total Affected"], errors="coerce").fillna(0),
            "damage_total_k": pd.to_numeric(raw["Total Damage ('000 US$)"], errors="coerce"),
            "damage_adjusted_k": pd.to_numeric(raw["Total Damage, Adjusted ('000 US$)"], errors="coerce"),
        }
    )

    prepared = prepared[
        prepared["region"].str.contains("Asia", case=False, na=False)
        & prepared["disaster_type"].isin(["Flood", "Storm"])
        & prepared["start_year"].between(2000, 2025, inclusive="both")
    ].copy()

    prepared["event_name"] = np.where(
        prepared["event_name"].str.len() > 0,
        prepared["event_name"],
        prepared["disaster_type"]
        + " in "
        + prepared["country"]
        + " ("
        + prepared["start_year"].astype(int).astype(str)
        + ")",
    )
    prepared["damage_k"] = prepared["damage_adjusted_k"]
    prepared["damage_k"] = prepared["damage_k"].fillna(prepared["damage_total_k"])
    prepared["damage_k"] = prepared["damage_k"].fillna(0).clip(lower=0)
    prepared["damage_musd"] = prepared["damage_k"] / 1000.0
    prepared["duration_days"] = _build_duration_days(prepared)
    prepared["start_month"] = prepared["start_month"].clip(lower=1, upper=12).astype(int)
    prepared["season"] = prepared["start_month"].apply(season_from_month)

    prepared = _filter_outliers(prepared)

    deaths_signal = _scaled_log_signal(prepared["total_deaths"])
    affected_signal = _scaled_log_signal(prepared["total_affected"])
    damage_signal = _scaled_log_signal(prepared["damage_k"])

    prepared["severity_score"] = SEVERITY_MAX * (
        0.30 * deaths_signal + 0.50 * affected_signal + 0.20 * damage_signal
    )
    prepared["historical_image_ratio"] = _build_historical_image_ratio(
        prepared,
        deaths_signal=deaths_signal,
        affected_signal=affected_signal,
        damage_signal=damage_signal,
    )
    prepared["severity_band"] = prepared["severity_score"].apply(severity_band_label)
    prepared["total_affected_log"] = np.log1p(prepared["total_affected"].clip(lower=0))

    country_stats = (
        prepared.groupby("country")
        .agg(
            country_event_count=("disno", "size"),
            country_mean_severity=("severity_score", "mean"),
            country_mean_affected=("total_affected", "mean"),
            country_mean_damage=("damage_musd", "mean"),
        )
        .reset_index()
    )
    prepared = prepared.merge(country_stats, on="country", how="left")
    prepared["country_mean_affected_log"] = np.log1p(prepared["country_mean_affected"].fillna(0).clip(lower=0))

    prepared = _resolve_map_coordinates(prepared)
    prepared = prepared.sort_values(["start_year", "severity_score"], ascending=[False, False]).reset_index(drop=True)
    return prepared


class DisasterSeverityPredictor:
    base_numeric_features = [
        "reference_year",
        "start_month",
        "duration_days",
        "magnitude",
        "total_affected_log",
        "country_event_count",
        "country_mean_severity",
        "country_mean_affected_log",
        "country_mean_damage",
    ]
    ratio_feature = "image_ratio_input"
    numeric_features = base_numeric_features + [ratio_feature]
    categorical_features = [
        "disaster_type",
        "disaster_subtype",
        "subregion",
        "country",
        "season",
    ]

    def __init__(self, historical_df: pd.DataFrame):
        self.historical_df = historical_df.copy()
        self.country_to_subregion = (
            self.historical_df.groupby("country")["subregion"].agg(lambda values: values.mode().iloc[0]).to_dict()
        )
        self.country_type_baseline = (
            self.historical_df.groupby(["country", "disaster_type"])["severity_score"].mean().to_dict()
        )
        self.global_defaults = {
            "country_event_count": float(self.historical_df["country_event_count"].median()),
            "country_mean_severity": float(self.historical_df["country_mean_severity"].median()),
            "country_mean_affected_log": float(self.historical_df["country_mean_affected_log"].median()),
            "country_mean_damage": float(self.historical_df["country_mean_damage"].median()),
            "magnitude": float(self.historical_df.loc[self.historical_df["magnitude"].notna(), "magnitude"].median()),
        }

        self.linear_pipeline: Optional[Pipeline] = None
        self.base_random_forest_pipeline: Optional[Pipeline] = None
        self.random_forest_pipeline: Optional[Pipeline] = None
        self.image_ratio_pipeline: Optional[Pipeline] = None
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.feature_importance: pd.DataFrame = pd.DataFrame()
        self.training_frame = self._build_training_frame()
        self.train()

    def _build_preprocessor(self, *, include_ratio: bool = True) -> ColumnTransformer:
        numeric_features = self.numeric_features if include_ratio else self.base_numeric_features
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

    def _build_training_frame(self) -> pd.DataFrame:
        frame = self.historical_df.copy()
        frame["reference_year"] = frame["start_year"].astype(int)
        frame["disaster_subtype"] = frame["disaster_subtype"].replace("", "Unspecified").fillna("Unspecified")
        frame["magnitude"] = pd.to_numeric(frame["magnitude"], errors="coerce")
        # Sort by year so TimeSeriesSplit folds are strictly chronological —
        # no future data leaks into earlier folds during cross-validation.
        return frame.sort_values("reference_year").reset_index(drop=True)

    def _build_ratio_estimator_template(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(include_ratio=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def _build_base_rf_template(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(include_ratio=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=180,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def _build_ratio_aware_rf_template(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(include_ratio=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=220,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def _evaluate_model(self, model_name: str, predictions: np.ndarray, targets: pd.Series) -> None:
        clipped_predictions = np.clip(predictions, 0, SEVERITY_MAX)
        self.model_metrics[model_name] = {
            "r2": float(r2_score(targets, clipped_predictions)),
            "mae": float(mean_absolute_error(targets, clipped_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, clipped_predictions))),
        }

    def _run_kfold_cv(
        self,
        pipeline_template: Pipeline,
        features: pd.DataFrame,
        target: pd.Series,
        model_name: str,
        n_splits: int = CV_SPLITS,
    ) -> Dict[str, float]:
        # TimeSeriesSplit ensures each validation fold only contains records
        # that come AFTER all training records in that fold — no future leakage.
        # Data must be sorted by year before splitting (done in _build_training_frame).
        tscv = TimeSeriesSplit(n_splits=n_splits)

        r2_scores = []
        mae_scores = []

        for train_idx, val_idx in tscv.split(features):
            x_train_fold = features.iloc[train_idx]
            y_train_fold = target.iloc[train_idx]
            x_val_fold = features.iloc[val_idx]
            y_val_fold = target.iloc[val_idx]

            fold_pipeline = clone(pipeline_template)
            fold_pipeline.fit(x_train_fold, y_train_fold)
            predictions = fold_pipeline.predict(x_val_fold)
            clipped_predictions = np.clip(predictions, 0, SEVERITY_MAX)

            r2_scores.append(r2_score(y_val_fold, clipped_predictions))
            mae_scores.append(mean_absolute_error(y_val_fold, clipped_predictions))

        return {
            "cv_r2_mean": float(np.mean(r2_scores)),
            "cv_r2_std": float(np.std(r2_scores)),
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
        }

    def _run_ratio_aware_cv(
        self,
        ratio_template: Pipeline,
        severity_template: Pipeline,
        base_features: pd.DataFrame,
        target: pd.Series,
        ratio_target: pd.Series,
        n_splits: int = CV_SPLITS,
    ) -> Dict[str, float]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        r2_scores = []
        mae_scores = []

        for train_idx, val_idx in tscv.split(base_features):
            x_train_base = base_features.iloc[train_idx].copy()
            x_val_base = base_features.iloc[val_idx].copy()
            y_train = target.iloc[train_idx]
            y_val = target.iloc[val_idx]
            ratio_train = ratio_target.iloc[train_idx]

            ratio_pipeline = clone(ratio_template)
            ratio_pipeline.fit(x_train_base, ratio_train)
            estimated_val_ratio = np.clip(
                ratio_pipeline.predict(x_val_base),
                0,
                IMAGE_RATIO_MAX,
            )

            x_train_ratio = x_train_base.copy()
            x_train_ratio[self.ratio_feature] = ratio_train.to_numpy()
            x_val_ratio = x_val_base.copy()
            x_val_ratio[self.ratio_feature] = estimated_val_ratio

            severity_pipeline = clone(severity_template)
            severity_pipeline.fit(x_train_ratio, y_train)
            predictions = severity_pipeline.predict(x_val_ratio)
            clipped_predictions = np.clip(predictions, 0, SEVERITY_MAX)

            r2_scores.append(r2_score(y_val, clipped_predictions))
            mae_scores.append(mean_absolute_error(y_val, clipped_predictions))

        return {
            "cv_r2_mean": float(np.mean(r2_scores)),
            "cv_r2_std": float(np.std(r2_scores)),
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
        }

    def _build_feature_importance(self) -> pd.DataFrame:
        if self.random_forest_pipeline is None:
            return pd.DataFrame()

        preprocessor = self.random_forest_pipeline.named_steps["preprocessor"]
        model = self.random_forest_pipeline.named_steps["model"]

        try:
            transformed_feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return pd.DataFrame()

        grouped_importance: Dict[str, float] = {}
        for feature_name, importance in zip(transformed_feature_names, model.feature_importances_):
            clean_name = feature_name.split("__", 1)[-1]
            if clean_name.startswith("reference_year"):
                label = "Reference year"
            elif clean_name.startswith("start_month"):
                label = "Start month"
            elif clean_name.startswith("duration_days"):
                label = "Duration"
            elif clean_name.startswith("magnitude"):
                label = "Magnitude"
            elif clean_name.startswith("total_affected_log"):
                label = "Affected population"
            elif clean_name.startswith("country_event_count"):
                label = "Country event count"
            elif clean_name.startswith("country_mean_severity"):
                label = "Country historical severity"
            elif clean_name.startswith("country_mean_affected_log"):
                label = "Country historical exposure"
            elif clean_name.startswith("country_mean_damage"):
                label = "Country historical damage"
            elif clean_name.startswith(self.ratio_feature):
                label = "Image damage ratio"
            elif clean_name.startswith("disaster_type_"):
                label = "Disaster type"
            elif clean_name.startswith("disaster_subtype_"):
                label = "Disaster subtype"
            elif clean_name.startswith("subregion_"):
                label = "Subregion"
            elif clean_name.startswith("country_"):
                label = "Country"
            elif clean_name.startswith("season_"):
                label = "Season"
            else:
                label = clean_name
            grouped_importance[label] = grouped_importance.get(label, 0.0) + float(importance)

        importance_frame = pd.DataFrame(
            [{"feature": key, "importance": value} for key, value in grouped_importance.items()]
        ).sort_values("importance", ascending=False)
        return importance_frame.reset_index(drop=True)

    def train(self) -> None:
        base_features = self.training_frame[self.base_numeric_features + self.categorical_features]
        target = self.training_frame["severity_score"]
        ratio_target = self.training_frame["historical_image_ratio"]

        # ── Temporal holdout: train on 2000-2020, test on 2021-2025 ──────
        # This is a more honest evaluation than a random split because disaster
        # data has temporal structure — a random split leaks future country
        # statistics into the training set.
        temporal_mask = self.training_frame["reference_year"] <= 2020
        train_idx = self.training_frame.index[temporal_mask]
        test_idx = self.training_frame.index[~temporal_mask]

        # Fall back to random 80/20 split if the temporal test set is too small
        # (fewer than 30 records), which can happen with very filtered datasets.
        if len(test_idx) < 30:
            train_idx, test_idx = train_test_split(
                self.training_frame.index,
                test_size=0.20,
                random_state=42,
            )

        x_train_base = base_features.loc[train_idx]
        x_test_base = base_features.loc[test_idx]
        y_train = target.loc[train_idx]
        y_test = target.loc[test_idx]
        ratio_train = ratio_target.loc[train_idx]
        ratio_test = ratio_target.loc[test_idx]

        self.holdout_info = {
            "strategy": "temporal" if len(test_idx) >= 30 else "random_80_20",
            "train_years": f"{int(self.training_frame.loc[train_idx, 'reference_year'].min())}–"
                           f"{int(self.training_frame.loc[train_idx, 'reference_year'].max())}",
            "test_years": f"{int(self.training_frame.loc[test_idx, 'reference_year'].min())}–"
                          f"{int(self.training_frame.loc[test_idx, 'reference_year'].max())}",
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        }

        linear_template = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(include_ratio=False)),
                ("model", LinearRegression()),
            ]
        )
        base_rf_template = self._build_base_rf_template()
        ratio_estimator_template = self._build_ratio_estimator_template()
        ratio_rf_template = self._build_ratio_aware_rf_template()

        linear_holdout = clone(linear_template)
        linear_holdout.fit(x_train_base, y_train)
        self._evaluate_model("Linear Regression", linear_holdout.predict(x_test_base), y_test)

        base_rf_holdout = clone(base_rf_template)
        base_rf_holdout.fit(x_train_base, y_train)
        self._evaluate_model("Random Forest (Past Data)", base_rf_holdout.predict(x_test_base), y_test)

        ratio_estimator_holdout = clone(ratio_estimator_template)
        ratio_estimator_holdout.fit(x_train_base, ratio_train)
        estimated_test_ratio = np.clip(
            ratio_estimator_holdout.predict(x_test_base),
            0,
            IMAGE_RATIO_MAX,
        )
        x_train_ratio = x_train_base.copy()
        x_train_ratio[self.ratio_feature] = ratio_train.to_numpy()
        x_test_ratio = x_test_base.copy()
        x_test_ratio[self.ratio_feature] = estimated_test_ratio

        ratio_rf_holdout = clone(ratio_rf_template)
        ratio_rf_holdout.fit(x_train_ratio, y_train)
        self._evaluate_model("Random Forest + Image Ratio", ratio_rf_holdout.predict(x_test_ratio), y_test)

        linear_cv = self._run_kfold_cv(linear_template, base_features, target, "Linear Regression")
        self.cv_results["Linear Regression"] = linear_cv
        self.model_metrics["Linear Regression"]["cv_r2"] = linear_cv["cv_r2_mean"]
        self.model_metrics["Linear Regression"]["cv_mae"] = linear_cv["cv_mae_mean"]
        
        base_rf_cv = self._run_kfold_cv(base_rf_template, base_features, target, "Random Forest (Past Data)")
        self.cv_results["Random Forest (Past Data)"] = base_rf_cv
        self.model_metrics["Random Forest (Past Data)"]["cv_r2"] = base_rf_cv["cv_r2_mean"]
        self.model_metrics["Random Forest (Past Data)"]["cv_mae"] = base_rf_cv["cv_mae_mean"]

        ratio_rf_cv = self._run_ratio_aware_cv(
            ratio_estimator_template,
            ratio_rf_template,
            base_features,
            target,
            ratio_target,
        )
        self.cv_results["Random Forest + Image Ratio"] = ratio_rf_cv
        self.model_metrics["Random Forest + Image Ratio"]["cv_r2"] = ratio_rf_cv["cv_r2_mean"]
        self.model_metrics["Random Forest + Image Ratio"]["cv_mae"] = ratio_rf_cv["cv_mae_mean"]

        full_ratio_features = base_features.copy()
        full_ratio_features[self.ratio_feature] = ratio_target.to_numpy()

        self.linear_pipeline = linear_template.fit(base_features, target)
        self.base_random_forest_pipeline = base_rf_template.fit(base_features, target)
        self.image_ratio_pipeline = ratio_estimator_template.fit(base_features, ratio_target)
        self.random_forest_pipeline = ratio_rf_template.fit(full_ratio_features, target)
        self.feature_importance = self._build_feature_importance()

    def _country_stats_for_prediction(self, country: str) -> Dict[str, float]:
        subset = self.historical_df[self.historical_df["country"] == country]
        if subset.empty:
            return self.global_defaults.copy()
        return {
            "country_event_count": float(subset["country_event_count"].median()),
            "country_mean_severity": float(subset["country_mean_severity"].median()),
            "country_mean_affected_log": float(subset["country_mean_affected_log"].median()),
            "country_mean_damage": float(subset["country_mean_damage"].median()),
            "magnitude": float(subset["magnitude"].dropna().median()) if subset["magnitude"].notna().any() else 0.0,
        }

    def build_scenario_frame(
        self,
        reference_year: int,
        disaster_type: str,
        country: str,
        disaster_subtype: str,
        start_month: int,
        duration_days: int,
        total_affected: int,
        magnitude: Optional[float] = None,
    ) -> pd.DataFrame:
        subregion = self.country_to_subregion.get(country, "Unknown")
        country_stats = self._country_stats_for_prediction(country)
        resolved_magnitude = country_stats["magnitude"] if magnitude in (None, 0) else float(magnitude)

        return pd.DataFrame(
            [
                {
                    "reference_year": int(reference_year),
                    "disaster_type": disaster_type,
                    "disaster_subtype": disaster_subtype or "Unspecified",
                    "subregion": subregion,
                    "country": country,
                    "season": season_from_month(int(start_month)),
                    "start_month": int(start_month),
                    "duration_days": max(1, int(duration_days)),
                    "magnitude": resolved_magnitude,
                    "total_affected_log": float(np.log1p(max(0, total_affected))),
                    "country_event_count": country_stats["country_event_count"],
                    "country_mean_severity": country_stats["country_mean_severity"],
                    "country_mean_affected_log": country_stats["country_mean_affected_log"],
                    "country_mean_damage": country_stats["country_mean_damage"],
                }
            ]
        )

    def estimate_image_ratio(self, scenario_frame: pd.DataFrame) -> float:
        if self.image_ratio_pipeline is None:
            return 0.0
        estimate = float(self.image_ratio_pipeline.predict(scenario_frame)[0])
        return float(np.clip(estimate, 0.0, IMAGE_RATIO_MAX))

    def analyze_satellite_image(self, image_bytes: bytes, disaster_type: str) -> Dict[str, object]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        image.thumbnail((512, 512))

        rgb = np.asarray(image).astype(np.float32) / 255.0
        analysis = _flood_water_analysis(rgb) if disaster_type == "Flood" else _storm_damage_analysis(rgb)
        detection_mask = analysis["mask"]
        detection_overlay = _build_detection_overlay(rgb, detection_mask, analysis["overlay_color"])
        
        sky_mask = (rgb[:, :, 2] >= 0.88) & ((rgb.mean(axis=2) - rgb.min(axis=2)) <= 0.15)
        kernel = np.ones((7, 7), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 127
        land_pixels = ~sky_mask
        land_count = land_pixels.sum()
        
        if land_count > 0:
            damage_ratio = detection_mask[land_pixels].mean()
        else:
            damage_ratio = float(detection_mask.mean())
        
        mask_coverage = float(detection_mask.mean())
        channel_means = rgb.mean(axis=(0, 1))
        return {
            "image_damage_score": round(float(analysis["image_damage_score"]), 1),
            "image_damage_ratio": round(float(damage_ratio * 100), 2),
            "analysis_confidence": float(analysis["analysis_confidence"]),
            "summary": analysis["summary"],
            "signals": analysis["signals"],
            "mask_coverage": round(mask_coverage * 100, 1),
            "coverage_label": analysis["coverage_label"],
            "score_label": analysis["score_label"],
            "overlay_caption": analysis["overlay_caption"],
            "detection_overlay": detection_overlay,
            "image_size": {"width": int(original_size[0]), "height": int(original_size[1])},
            "scene_profile": {
                "red_mean": round(float(channel_means[0] * 255), 1),
                "green_mean": round(float(channel_means[1] * 255), 1),
                "blue_mean": round(float(channel_means[2] * 255), 1),
            },
        }

    def find_similar_events(
        self,
        reference_year: int,
        disaster_type: str,
        country: str,
        start_month: int,
        duration_days: int,
        total_affected: int,
        magnitude: Optional[float] = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        candidates = self.historical_df[self.historical_df["disaster_type"] == disaster_type].copy()
        if candidates.empty:
            return pd.DataFrame()

        target_affected_log = float(np.log1p(max(0, total_affected)))
        magnitude_value = 0.0 if magnitude in (None, 0) else float(magnitude)
        candidates["distance"] = (
            1.60 * np.abs(candidates["total_affected_log"] - target_affected_log)
            + 0.04 * np.abs(candidates["duration_days"] - duration_days)
            + 0.05 * np.abs(candidates["start_year"] - reference_year)
            + 0.07 * np.abs(candidates["start_month"] - start_month)
            + 0.06 * np.abs(candidates["magnitude"].fillna(0) - magnitude_value)
        )
        candidates.loc[candidates["country"] == country, "distance"] *= 0.55
        candidates.loc[
            candidates["subregion"] == self.country_to_subregion.get(country, "Unknown"),
            "distance",
        ] *= 0.80

        similar = (
            candidates.sort_values("distance")
            .head(top_n)[
                [
                    "start_year",
                    "country",
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
                    "disaster_type": "Type",
                    "disaster_subtype": "Subtype",
                    "event_name": "Event",
                    "severity_score": "Severity Score",
                    "total_deaths": "Deaths",
                    "total_affected": "Affected",
                    "damage_musd": "Damage (M US$)",
                }
            )
        )
        return similar.reset_index(drop=True)

    def predict(
        self,
        reference_year: int,
        disaster_type: str,
        country: str,
        disaster_subtype: str,
        start_month: int,
        duration_days: int,
        total_affected: int,
        magnitude: Optional[float] = None,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, object]:
        if (
            self.linear_pipeline is None
            or self.base_random_forest_pipeline is None
            or self.random_forest_pipeline is None
            or self.image_ratio_pipeline is None
        ):
            raise ValueError("Models are not trained.")

        scenario_base = self.build_scenario_frame(
            reference_year=reference_year,
            disaster_type=disaster_type,
            country=country,
            disaster_subtype=disaster_subtype,
            start_month=start_month,
            duration_days=duration_days,
            total_affected=total_affected,
            magnitude=magnitude,
        )

        linear_score = float(np.clip(self.linear_pipeline.predict(scenario_base)[0], 0, SEVERITY_MAX))
        base_rf_score = float(
            np.clip(self.base_random_forest_pipeline.predict(scenario_base)[0], 0, SEVERITY_MAX)
        )
        estimated_image_ratio = self.estimate_image_ratio(scenario_base)

        country_stats = self._country_stats_for_prediction(country)
        country_mean_affected = np.expm1(country_stats["country_mean_affected_log"])
        affected_ratio = float(total_affected) / max(country_mean_affected, 1)

        boost_factor = 1.0
        if affected_ratio > 3.0:
            boost_factor = min(1.15, 1.0 + (affected_ratio - 3.0) * 0.05)
        elif affected_ratio > 1.5:
            boost_factor = 1.0 + (affected_ratio - 1.5) * 0.10

        image_result = None
        image_damage_score: Optional[float] = None
        image_damage_ratio: Optional[float] = None
        image_ratio_used = estimated_image_ratio
        image_ratio_source = "Historical estimate"
        
        if image_bytes:
            image_result = self.analyze_satellite_image(image_bytes, disaster_type)
            image_damage_score = float(image_result.get("image_damage_score", 0.0))
            image_damage_ratio = float(image_result.get("image_damage_ratio", 0.0))
            image_ratio_used = image_damage_ratio
            image_ratio_source = "Uploaded image"

        scenario_with_ratio = scenario_base.copy()
        scenario_with_ratio[self.ratio_feature] = image_ratio_used
        ratio_rf_score = float(
            np.clip(self.random_forest_pipeline.predict(scenario_with_ratio)[0], 0, SEVERITY_MAX)
        )

        base_rf_score = min(SEVERITY_MAX, base_rf_score * boost_factor)
        ratio_rf_score = min(SEVERITY_MAX, ratio_rf_score * boost_factor)

        fusion_decision = decide_image_fusion(ratio_rf_score, image_result)

        if image_result and image_damage_score is not None:
            # ── Confidence-weighted bidirectional blend ────────────────────
            # Weight w is derived from image confidence and fusion strength.
            # This replaces the old hard gate + one-directional floor:
            #   OLD: image merged only if RF+image > non-image baseline
            #         → systematic upward bias; image could never lower score
            #   NEW: w scales continuously with confidence; image can revise
            #         severity in either direction, but only as much as its
            #         confidence warrants.
            #
            # w = confidence_ratio × fusion_strength, clamped to [0.10, 0.45]
            #   - At 80% confidence + high fusion: w ≈ 0.40 → strong image pull
            #   - At 25% confidence + low fusion:  w ≈ 0.10 → image barely moves score
            #   - No floor: if image says 4.0 and model says 7.0 with w=0.30,
            #     final = 0.70×7.0 + 0.30×4.0 = 6.1  (image can reduce the score)
            image_confidence_ratio = fusion_decision.get("image_confidence", 50.0) / 100.0
            fusion_strength = fusion_decision.get("fusion_ratio", 0.5)
            image_blend_weight = float(
                np.clip(image_confidence_ratio * fusion_strength, 0.10, 0.45)
            )

            hybrid_score = (
                (1.0 - image_blend_weight) * ratio_rf_score
                + image_blend_weight * image_damage_score
            )
            weights = {
                "RF + Image Ratio": round(1.0 - image_blend_weight, 2),
                "Image-only signal": round(image_blend_weight, 2),
            }
            all_scores = [base_rf_score, ratio_rf_score, image_damage_score]
        else:
            hybrid_score = ratio_rf_score
            weights = {"RF + Historical Ratio": 1.0}
            all_scores = [base_rf_score, ratio_rf_score]

        model_spread = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
        ratio_rf_mae = self.model_metrics["Random Forest + Image Ratio"]["mae"]
        hybrid_mae = ratio_rf_mae * max(weights.values())
        if image_result and image_damage_score is not None:
            hybrid_mae += weights.get("Image-only signal", 0.0) * 0.9
            confidence_base = 88.0 + min(fusion_decision.get("image_confidence", 0.0) / 20.0, 4.0)
        else:
            confidence_base = 83.0

        interval_radius = min(2.2, hybrid_mae + model_spread * 0.45)
        lower_bound = float(np.clip(hybrid_score - interval_radius, 0, SEVERITY_MAX))
        upper_bound = float(np.clip(hybrid_score + interval_radius, 0, SEVERITY_MAX))

        confidence = float(np.clip(confidence_base - model_spread * 6.0, 60.0, 97.0))

        baseline_key = (country, disaster_type)
        type_mask = self.historical_df["disaster_type"] == disaster_type
        baseline = float(
            self.country_type_baseline.get(baseline_key, self.historical_df.loc[type_mask, "severity_score"].mean())
        )

        similar_events = self.find_similar_events(
            reference_year=reference_year,
            disaster_type=disaster_type,
            country=country,
            start_month=start_month,
            duration_days=duration_days,
            total_affected=total_affected,
            magnitude=magnitude,
        )

        drivers: List[str] = []
        if affected_ratio > 3.0:
            drivers.append(f"EXTREME: affected population ({affected_ratio:.1f}x country's average) significantly boosts severity")
        elif affected_ratio > 1.5:
            drivers.append(f"affected population ({affected_ratio:.1f}x average) is above the country's historical norm")
        if duration_days >= 14:
            drivers.append("long event duration increases severity")
        if hybrid_score >= baseline + 0.8:
            drivers.append("the scenario sits above the country's historical baseline")
        if image_result:
            drivers.append(
                f"uploaded image ratio ({image_ratio_used:.1f}%) replaced the historical prior ({estimated_image_ratio:.1f}%)"
            )
            if fusion_decision["use_image"]:
                drivers.append("satellite image signal strengthened the ratio-aware Random Forest estimate")
            else:
                drivers.append("satellite image was analyzed, but its influence stayed limited")
        if not drivers:
            drivers.append("historical patterns keep the scenario near baseline")

        if image_result and image_damage_score is not None and image_damage_ratio is not None:
            component_scores = {
                "RF (Past Data)": round(base_rf_score, 1),
                "RF + Image Ratio": round(ratio_rf_score, 1),
                "Image-only signal": round(image_damage_score, 1),
            }
            drivers.append(f"Image damage ratio: {image_damage_ratio:.1f}% with image-only score {image_damage_score:.1f}/10")
        else:
            component_scores = {
                "RF (Past Data)": round(base_rf_score, 1),
                "RF + Historical Ratio": round(ratio_rf_score, 1),
            }
            drivers.append(f"No satellite image uploaded - using historical image-ratio prior ({estimated_image_ratio:.1f}%)")

        return {
            "hybrid_score": round(float(np.clip(hybrid_score, 0, SEVERITY_MAX)), 1),
            "prediction_mode": "RF + Uploaded Image Ratio" if image_result else "RF + Historical Ratio",
            "severity_band": severity_band_label(float(hybrid_score)),
            "confidence": round(confidence, 1),
            "prediction_range": {"lower": round(lower_bound, 1), "upper": round(upper_bound, 1)},
            "component_scores": component_scores,
            "component_weights": weights,
            "linear_score": round(linear_score, 1),
            "base_rf_score": round(base_rf_score, 1),
            "ratio_rf_score": round(ratio_rf_score, 1),
            "estimated_image_ratio": round(estimated_image_ratio, 1),
            "image_ratio_used": round(image_ratio_used, 1),
            "image_ratio_source": image_ratio_source,
            "image_only_damage_score": round(image_damage_score, 1) if image_damage_score is not None else None,
            "image_damage_ratio": image_damage_ratio if image_damage_ratio is not None else None,
            "country_baseline": round(baseline, 1),
            "difference_from_baseline": round(float(hybrid_score - baseline), 1),
            "scenario": {
                "reference_year": int(reference_year),
                "disaster_type": disaster_type,
                "disaster_subtype": disaster_subtype,
                "country": country,
                "subregion": self.country_to_subregion.get(country, "Unknown"),
                "start_month": int(start_month),
                "duration_days": int(duration_days),
                "total_affected": int(total_affected),
                "magnitude": None if magnitude in (None, 0) else float(magnitude),
            },
            "drivers": drivers,
            "similar_events": similar_events,
            "image_analysis": image_result,
            "fusion_decision": fusion_decision,
        }

    @property
    def available_countries(self) -> List[str]:
        return sorted(self.historical_df["country"].dropna().unique().tolist())

    def available_subtypes(self, disaster_type: str) -> List[str]:
        subtypes = (
            self.historical_df.loc[self.historical_df["disaster_type"] == disaster_type, "disaster_subtype"]
            .fillna("Unspecified")
            .replace("", "Unspecified")
            .value_counts()
            .index.tolist()
        )
        return subtypes[:12] if subtypes else ["Unspecified"]

    def calibrate_image_analysis(self) -> pd.DataFrame:
        """
        Run the flood and storm detectors on synthetic reference images and
        return a calibration report.

        Reference images are generated programmatically so no external files
        are needed. Each case has a known expected outcome so the report
        immediately shows whether the heuristic detector is systematically off.

        Cases
        -----
        1. Solid blue              → flood: HIGH water coverage expected
        2. Solid green             → flood: LOW (vegetation, not water)
        3. Solid white             → flood: LOW (cloud/sky exclusion)
        4. Blue top / green bottom → flood: MODERATE (~50% water)
        5. Muddy brown             → flood: MODERATE-HIGH (sediment water)
        6. Solid gray              → storm: LOW damage signal
        7. Random pixel noise      → storm: MODERATE (high edge density)
        """
        import io

        def _solid(rgb_float, size: int = 128) -> bytes:
            arr = np.clip(np.full((size, size, 3), rgb_float, dtype=np.float32) * 255, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        def _half(top_rgb, bot_rgb, size: int = 128) -> bytes:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[: size // 2] = [int(v * 255) for v in top_rgb]
            arr[size // 2 :] = [int(v * 255) for v in bot_rgb]
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        def _noise(size: int = 128) -> bytes:
            arr = (np.random.default_rng(42).random((size, size, 3)) * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        cases = [
            ("Solid blue (clear water)",      "Flood", _solid((0.10, 0.30, 0.85)),               "HIGH"),
            ("Solid green (vegetation)",       "Flood", _solid((0.10, 0.65, 0.15)),               "LOW"),
            ("Solid white (cloud/sky)",        "Flood", _solid((0.97, 0.97, 0.97)),               "LOW"),
            ("Blue top / green bottom",        "Flood", _half((0.10, 0.30, 0.85), (0.10, 0.65, 0.15)), "MODERATE"),
            ("Muddy brown (sediment water)",   "Flood", _solid((0.55, 0.42, 0.25)),               "MODERATE-HIGH"),
            ("Solid gray (no damage cues)",    "Storm", _solid((0.50, 0.50, 0.50)),               "LOW"),
            ("Random noise (high edge)",       "Storm", _noise(),                                  "MODERATE"),
        ]

        def _bucket(pct: float, expected: str) -> str:
            if pct >= 30:
                return "HIGH"
            if pct >= 10:
                return "MODERATE-HIGH" if expected == "MODERATE-HIGH" else "MODERATE"
            if pct >= 3:
                return "LOW-MODERATE"
            return "LOW"

        rows = []
        for label, dtype, img_bytes, expected in cases:
            result = self.analyze_satellite_image(img_bytes, dtype)
            detected_pct = float(result["mask_coverage"])
            detected = _bucket(detected_pct, expected)
            passed = detected == expected or (
                expected == "MODERATE-HIGH" and detected in ("MODERATE", "HIGH", "MODERATE-HIGH")
            ) or (
                expected == "MODERATE" and detected in ("MODERATE", "MODERATE-HIGH", "LOW-MODERATE")
            )
            rows.append({
                "Case": label,
                "Type": dtype,
                "Expected": expected,
                "Detected coverage (%)": round(detected_pct, 1),
                "Detected label": detected,
                "Image score (/10)": result["image_damage_score"],
                "Confidence (%)": round(float(result["analysis_confidence"]), 1),
                "Pass": "YES" if passed else "NO",
            })
        return pd.DataFrame(rows)

    def model_quality_table(self) -> pd.DataFrame:
        holdout = getattr(self, "holdout_info", {})
        test_label = holdout.get("test_years", "20% holdout")
        strategy = holdout.get("strategy", "random_80_20")
        holdout_col = (
            f"Holdout R2 ({test_label})"
            if strategy == "temporal"
            else "Holdout R2 (random 20%)"
        )
        holdout_mae_col = (
            f"Holdout MAE ({test_label})"
            if strategy == "temporal"
            else "Holdout MAE (random 20%)"
        )
        rows = []
        for model_name, metrics in self.model_metrics.items():
            row = {
                "Model": model_name,
                holdout_col: round(metrics["r2"], 3),
                holdout_mae_col: round(metrics["mae"], 2),
                "RMSE (/10)": round(metrics["rmse"], 2),
            }
            if "cv_r2" in metrics:
                row[f"TS-CV R2 ({CV_SPLITS}-fold)"] = (
                    f"{metrics['cv_r2']:.3f} +/- {self.cv_results.get(model_name, {}).get('cv_r2_std', 0):.3f}"
                )
            if "cv_mae" in metrics:
                row[f"TS-CV MAE ({CV_SPLITS}-fold, /10)"] = (
                    f"{metrics['cv_mae']:.2f} +/- {self.cv_results.get(model_name, {}).get('cv_mae_std', 0):.2f}"
                )
            rows.append(row)
        return pd.DataFrame(rows)


def decide_image_fusion(model_score: float, image_result: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not image_result:
        return {
            "use_image": False,
            "fusion_ratio": 0.0,
            "agreement_ratio": 0.0,
            "reason": "No satellite image was uploaded, so the final severity uses the Random Forest plus a ratio estimated from past events.",
        }

    image_score = float(image_result["image_damage_score"])
    image_confidence = float(image_result.get("analysis_confidence", 0.0))
    mask_coverage = float(image_result.get("mask_coverage", 0.0)) / 100.0
    score_gap = abs(model_score - image_score)

    agreement_ratio = float(np.clip(1.0 - (score_gap / 4.0), 0.0, 1.0))
    confidence_ratio = float(np.clip(image_confidence / 100.0, 0.0, 1.0))
    coverage_ratio = float(np.clip(mask_coverage / 0.10, 0.0, 1.0))
    
    agreement_boost = 1.0 if score_gap <= 2.0 else 0.8 if score_gap <= 3.0 else 0.5
    
    fusion_ratio = float(
        np.clip(
            0.40 * agreement_ratio * agreement_boost + 
            0.35 * confidence_ratio + 
            0.25 * coverage_ratio,
            0.0,
            1.0,
        )
    )
    
    use_image = bool(fusion_ratio >= 0.35 and image_confidence >= 40.0 and score_gap <= 3.5)

    if use_image:
        direction = "raises" if image_score > model_score else "lowers"
        reason = (
            f"Strong satellite-image signal (confidence {image_confidence:.0f}%). "
            f"Image score ({image_score:.1f}/10) {direction} the final severity estimate. "
            f"Image blended with weight proportional to confidence."
        )
    elif image_confidence < 40.0:
        reason = (
            f"Weak image signal (confidence {image_confidence:.0f}%). "
            f"Image still blended at reduced weight — it can raise or lower the score."
        )
    elif score_gap > 3.5:
        reason = (
            f"Image score ({image_score:.1f}/10) diverges from model ({model_score:.1f}/10) "
            f"by {score_gap:.1f} points. Partial blend applied at reduced weight."
        )
    else:
        reason = "Image analysis complete. Confidence-weighted blend applied — image may raise or lower final score."

    return {
        "use_image": use_image,
        "fusion_ratio": round(fusion_ratio, 3),
        "agreement_ratio": round(agreement_ratio, 3),
        "confidence_ratio": round(confidence_ratio, 3),
        "coverage_ratio": round(coverage_ratio, 3),
        "score_gap": round(score_gap, 1),
        "image_confidence": round(image_confidence, 1),
        "reason": reason,
    }


def get_legacy_model_summary(model_path: str) -> Dict[str, str]:
    path = Path(model_path)
    if not path.exists():
        return {"available": False, "message": "Legacy MLR artifact not found."}

    try:
        model = joblib.load(path)
        return {
            "available": True,
            "estimator": type(model).__name__,
            "message": "Legacy notebook model loaded successfully.",
        }
    except Exception as exc:
        return {"available": False, "message": f"Legacy MLR artifact could not be loaded: {exc}"}


def filter_historical_data(
    dataframe: pd.DataFrame,
    disaster_types: List[str],
    year_range: Tuple[int, int],
    country: str = "All",
    subregions: Optional[List[str]] = None,
) -> pd.DataFrame:
    filtered = dataframe[dataframe["start_year"].between(year_range[0], year_range[1], inclusive="both")].copy()
    if disaster_types:
        filtered = filtered[filtered["disaster_type"].isin(disaster_types)]
    if country != "All":
        filtered = filtered[filtered["country"] == country]
    if subregions:
        filtered = filtered[filtered["subregion"].isin(subregions)]
    return filtered.reset_index(drop=True)


def apply_chart_theme(fig, *, show_x_grid: bool = False, show_y_grid: bool = True):
    fig.update_layout(
        paper_bgcolor=APP_PALETTE["chart_surface"],
        plot_bgcolor=APP_PALETTE["chart_surface"],
        font=dict(color=APP_PALETTE["text"], size=13),
        title_font=dict(color=APP_PALETTE["text"], size=18),
        hoverlabel=dict(
            bgcolor=APP_PALETTE["surface"],
            bordercolor=APP_PALETTE["border"],
            font_color=APP_PALETTE["text"],
        ),
    )
    fig.update_xaxes(
        showgrid=show_x_grid,
        gridcolor=APP_PALETTE["grid"],
        zeroline=False,
        linecolor=APP_PALETTE["border"],
        tickfont=dict(color=APP_PALETTE["text"]),
        title_font=dict(color=APP_PALETTE["text"]),
    )
    fig.update_yaxes(
        showgrid=show_y_grid,
        gridcolor=APP_PALETTE["grid"],
        zeroline=False,
        linecolor=APP_PALETTE["border"],
        tickfont=dict(color=APP_PALETTE["text"]),
        title_font=dict(color=APP_PALETTE["text"]),
    )
    return fig


def build_yearly_events_chart(filtered_df: pd.DataFrame):
    yearly = filtered_df.groupby(["start_year", "disaster_type"]).agg(events=("disno", "size")).reset_index()
    fig = px.bar(
        yearly,
        x="start_year",
        y="events",
        color="disaster_type",
        barmode="group",
        color_discrete_map=DISASTER_COLORS,
        labels={"start_year": "Year", "events": "Events", "disaster_type": "Type"},
        title="Events by year",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return apply_chart_theme(fig, show_x_grid=False, show_y_grid=True)


def build_severity_distribution_chart(filtered_df: pd.DataFrame):
    fig = px.histogram(
        filtered_df,
        x="severity_score",
        color="disaster_type",
        nbins=10,
        opacity=0.80,
        color_discrete_map=DISASTER_COLORS,
        labels={"severity_score": "Severity score (/10)", "count": "Events"},
        title="Severity distribution",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return apply_chart_theme(fig, show_x_grid=False, show_y_grid=True)


def build_country_summary_chart(filtered_df: pd.DataFrame):
    country_summary = (
        filtered_df.groupby("country")
        .agg(events=("disno", "size"), avg_severity=("severity_score", "mean"), affected=("total_affected", "sum"))
        .sort_values(["events", "avg_severity"], ascending=[False, False])
        .head(12)
        .reset_index()
    )
    fig = px.bar(
        country_summary,
        x="country",
        y="events",
        color="events",
        color_continuous_scale=EVENT_COUNT_COLOR_SCALE,
        hover_data={"affected": ":,", "avg_severity": ":.1f", "events": ":,"},
        labels={"country": "Country", "events": "Events", "avg_severity": "Avg severity (/10)"},
        title="Top countries in the filtered view",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return apply_chart_theme(fig, show_x_grid=False, show_y_grid=True)


def build_feature_importance_chart(feature_importance_df: pd.DataFrame, top_n: int = 8):
    if feature_importance_df.empty:
        fig = px.bar(
            x=[0], y=["No data"], text=["No data"],
            labels={"x": "Feature Importance", "y": ""},
            title="Feature Importance (Why this prediction?)"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        return apply_chart_theme(fig, show_x_grid=True, show_y_grid=False)
    
    top_features = feature_importance_df.head(top_n).copy()
    top_features = top_features.sort_values("importance", ascending=True)
    
    fig = px.bar(
        top_features,
        y="feature",
        x="importance",
        orientation="h",
        color="importance",
        color_continuous_scale=FEATURE_IMPORTANCE_SCALE,
        text="importance",
        labels={"importance": "Importance Score", "feature": ""},
        title="Why this prediction? (Feature Importance)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"}
    )
    return apply_chart_theme(fig, show_x_grid=True, show_y_grid=False)


def build_severity_gauge(score: float) -> dict:
    if score >= 70:
        color = "#dc3545"
        level = "HIGH"
        emoji = "🔴"
    elif score >= 40:
        color = "#ffc107"
        level = "MEDIUM"
        emoji = "🟡"
    else:
        color = "#28a745"
        level = "LOW"
        emoji = "🟢"
    
    return {
        "score": score,
        "color": color,
        "level": level,
        "emoji": emoji,
        "gauge_value": score / 100,
    }


def build_confidence_badge(confidence: float, rf_score: float, linear_score: float) -> dict:
    model_spread = abs(rf_score - linear_score)
    
    if model_spread <= 10 and confidence >= 80:
        level = "HIGH"
        color = "#28a745"
        description = "Models strongly agree"
    elif model_spread <= 20 and confidence >= 60:
        level = "MEDIUM"
        color = "#ffc107"
        description = "Models moderately agree"
    else:
        level = "LOW"
        color = "#dc3545"
        description = "Models disagree - use with caution"
    
    return {
        "confidence": confidence,
        "level": level,
        "color": color,
        "description": description,
        "model_spread": round(model_spread, 1),
    }


def build_map_figure(filtered_df: pd.DataFrame):
    map_ready = filtered_df.dropna(subset=["map_latitude", "map_longitude"]).copy()
    map_ready = map_ready[
        map_ready["map_latitude"].between(-10, 55, inclusive="both")
        & map_ready["map_longitude"].between(40, 150, inclusive="both")
    ].copy()
    fig = px.scatter_geo(
        map_ready,
        lat="map_latitude",
        lon="map_longitude",
        color="disaster_type",
        size="severity_score",
        size_max=18,
        hover_name="event_name",
        hover_data={
            "country": True,
            "subregion": True,
            "start_year": True,
            "severity_score": ":.1f",
            "map_latitude": False,
            "map_longitude": False,
        },
        color_discrete_map=DISASTER_COLORS,
        title="Asia flood and storm events",
    )
    fig.update_geos(
        scope="asia",
        showcountries=True,
        showland=True,
        landcolor="#f4efe7",
        showocean=True,
        oceancolor="#d9eef7",
        lataxis_range=[-10, 55],
        lonaxis_range=[40, 150],
    )
    fig.update_layout(
        height=800,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
    )
    return fig


def build_component_chart(prediction: Dict[str, object]):
    chart_frame = pd.DataFrame(
        [
            {
                "Component": name,
                "Score": score,
                "Blend weight (%)": round(float(prediction["component_weights"].get(name, 0.0)) * 100, 1),
            }
            for name, score in prediction["component_scores"].items()
        ]
    )
    fig = px.bar(
        chart_frame,
        x="Component",
        y="Score",
        color="Component",
        text="Blend weight (%)",
        labels={"Score": "Model estimate", "Component": ""},
        title="Severity model estimates",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), showlegend=False, yaxis_range=[0, 100])
    return fig


def build_severity_gauge(score: float) -> dict:
    if score >= GAUGE_HIGH_THRESHOLD:
        color = APP_PALETTE["high"]
        level = "HIGH"
        badge = "HIGH"
    elif score >= GAUGE_MEDIUM_THRESHOLD:
        color = APP_PALETTE["medium"]
        level = "MEDIUM"
        badge = "MEDIUM"
    else:
        color = APP_PALETTE["low"]
        level = "LOW"
        badge = "LOW"

    return {
        "score": score,
        "color": color,
        "level": level,
        "emoji": badge,
        "gauge_value": score / SEVERITY_MAX,
    }


def build_confidence_badge(confidence: float, rf_score: float, linear_score: float) -> dict:
    model_spread = abs(rf_score - linear_score)

    if model_spread <= 1.0 and confidence >= 80:
        level = "HIGH"
        color = APP_PALETTE["low"]
        description = "Models strongly agree"
    elif model_spread <= 2.0 and confidence >= 60:
        level = "MEDIUM"
        color = APP_PALETTE["medium"]
        description = "Models moderately agree"
    else:
        level = "LOW"
        color = APP_PALETTE["high"]
        description = "Models disagree - use with caution"

    return {
        "confidence": confidence,
        "level": level,
        "color": color,
        "description": description,
        "model_spread": round(model_spread, 1),
    }


def build_map_figure(filtered_df: pd.DataFrame):
    map_ready = filtered_df.dropna(subset=["map_latitude", "map_longitude"]).copy()
    map_ready = map_ready[
        map_ready["map_latitude"].between(-10, 55, inclusive="both")
        & map_ready["map_longitude"].between(40, 150, inclusive="both")
    ].copy()
    fig = px.scatter_geo(
        map_ready,
        lat="map_latitude",
        lon="map_longitude",
        color="disaster_type",
        size="severity_score",
        size_max=18,
        hover_name="event_name",
        hover_data={
            "country": True,
            "subregion": True,
            "start_year": True,
            "severity_score": ":.1f",
            "map_latitude": False,
            "map_longitude": False,
        },
        color_discrete_map=DISASTER_COLORS,
        title="Asia flood and storm events",
    )
    fig.update_geos(
        scope="asia",
        showcountries=True,
        showland=True,
        landcolor=APP_PALETTE["land"],
        showocean=True,
        oceancolor=APP_PALETTE["ocean"],
        lataxis_range=[-10, 55],
        lonaxis_range=[40, 150],
    )
    fig.update_layout(
        height=800,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        paper_bgcolor=APP_PALETTE["chart_surface"],
        font=dict(color=APP_PALETTE["text"]),
    )
    fig.update_geos(bgcolor=APP_PALETTE["chart_surface"])
    return fig


def build_component_chart(prediction: Dict[str, object]):
    chart_frame = pd.DataFrame(
        [
            {
                "Component": name,
                "Score": score,
                "Blend weight (%)": round(float(prediction["component_weights"].get(name, 0.0)) * 100, 1),
            }
            for name, score in prediction["component_scores"].items()
        ]
    )
    fig = px.bar(
        chart_frame,
        x="Component",
        y="Score",
        color="Component",
        text="Blend weight (%)",
        color_discrete_sequence=[APP_PALETTE["accent"], APP_PALETTE["flood"], APP_PALETTE["storm"]],
        labels={"Score": "Model estimate (/10)", "Component": ""},
        title="Severity model estimates",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        yaxis_range=[0, SEVERITY_MAX],
    )
    return apply_chart_theme(fig, show_x_grid=False, show_y_grid=True)


class MLRSeverityPredictor:
    """
    Multiple Linear Regression predictor restricted to the exact feature columns:

    Index(['Total Affected', 'Disaster_Duration_Days', 'Subregion_Eastern Asia',
           'Subregion_South-eastern Asia', 'Subregion_Southern Asia',
           'Subregion_Western Asia', 'Disaster Type_Storm'], dtype='str')
    """

    feature_columns = [
        "Total Affected",
        "Disaster_Duration_Days",
        "Subregion_Eastern Asia",
        "Subregion_South-eastern Asia",
        "Subregion_Southern Asia",
        "Subregion_Western Asia",
        "Disaster Type_Storm",
    ]

    def __init__(self, historical_df: pd.DataFrame):
        self.historical_df = historical_df.copy()
        self.country_to_subregion = (
            self.historical_df.groupby("country")["subregion"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict()
        )
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        self.metrics: Dict[str, float] = {}
        self._train()

    @property
    def available_countries(self) -> List[str]:
        return sorted(self.historical_df["country"].dropna().unique().tolist())

    def _design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=frame.index)
        affected = pd.to_numeric(frame["total_affected"], errors="coerce").fillna(0).clip(lower=0)
        duration = pd.to_numeric(frame["duration_days"], errors="coerce").fillna(1).clip(lower=1)
        # Keep the SAME 7 feature columns, but transform values for stability.
        x["Total Affected"] = np.log1p(affected)
        # Encode an interaction-like effect without adding new columns:
        # duration impact grows with affected population.
        x["Disaster_Duration_Days"] = np.log1p(duration) * (1.0 + np.log1p(affected) / 6.0)

        subregion = frame["subregion"].fillna("Unknown").astype(str)
        x["Subregion_Eastern Asia"] = (subregion == "Eastern Asia").astype(int)
        x["Subregion_South-eastern Asia"] = (subregion == "South-eastern Asia").astype(int)
        x["Subregion_Southern Asia"] = (subregion == "Southern Asia").astype(int)
        x["Subregion_Western Asia"] = (subregion == "Western Asia").astype(int)

        dtype = frame["disaster_type"].fillna("").astype(str)
        x["Disaster Type_Storm"] = (dtype == "Storm").astype(int)

        return x[self.feature_columns]

    def _train(self) -> None:
        x = self._design_matrix(self.historical_df)
        y = pd.to_numeric(self.historical_df["severity_score"], errors="coerce").fillna(0).clip(0, SEVERITY_MAX)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.20,
            random_state=42,
        )
        self.model.fit(x_train, y_train)
        preds = np.clip(self.model.predict(x_test), 0, SEVERITY_MAX)
        self.metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }

    def model_quality_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Model": "Multiple Linear Regression",
                    "R2": round(self.metrics.get("r2", 0.0), 3),
                    "MAE (/10)": round(self.metrics.get("mae", 0.0), 2),
                    "RMSE (/10)": round(self.metrics.get("rmse", 0.0), 2),
                }
            ]
        )

    def predict_score(
        self,
        *,
        disaster_type: str,
        subregion: str,
        total_affected: int,
        duration_days: int,
    ) -> float:
        row = pd.DataFrame(
            [
                {
                    "disaster_type": disaster_type,
                    "subregion": subregion,
                    "total_affected": int(max(0, total_affected)),
                    "duration_days": int(max(1, duration_days)),
                    "severity_score": 0.0,
                }
            ]
        )
        x = self._design_matrix(row)
        return float(np.clip(self.model.predict(x)[0], 0, SEVERITY_MAX))

