"""
Compatibility wrapper.

The Streamlit UI has been moved to `app.py` and the backend logic to `backend.py`.
Run the app with: `streamlit run app.py`.
"""

from __future__ import annotations

from app import main


def _cleanup_mask(
    mask: np.ndarray,
    *,
    open_size: int = 5,
    close_size: int = 11,
    min_area_ratio: float = 0.0006,
) -> np.ndarray:
    """Remove small artifacts from binary image masks."""
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
    """Check whether a connected component reaches the image boundary."""
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
    """
    Detect flood water in blue, gray, and muddy scenes using color plus texture.

    The reference flood set includes clean water, sediment-heavy water, and
    washed-out gray floodwater, so the mask uses multiple color families and
    then keeps only smooth, water-like connected regions.
    """
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

    seed_mask = (blue_seed | muddy_seed | dark_seed) & smooth & ~vegetation & ~bright_cloud
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

    blue_candidate = (
        (hue >= 78) & (hue <= 145) & (saturation >= 0.05) & (value >= 0.10)
    )
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
        (
            ((blue_candidate | muddy_candidate | dark_candidate) & smooth)
            | (gray_candidate & very_smooth & expanded_seed)
        )
        & ~vegetation
        & ~bright_cloud
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

    mask_coverage = float(mask.mean())
    seed_coverage = float(seed_mask.mean())
    smooth_surface_ratio = (
        float(np.mean(very_smooth[mask])) if mask.any() else 0.0
    )
    blue_ratio = float(np.mean(blue_candidate[mask])) if mask.any() else 0.0
    gray_ratio = float(np.mean(gray_candidate[mask])) if mask.any() else 0.0
    muddy_ratio = float(np.mean(muddy_candidate[mask])) if mask.any() else 0.0
    open_water_signal = float(np.clip(mask_coverage / 0.18, 0.0, 1.0))
    seed_signal = float(np.clip(seed_coverage / 0.04, 0.0, 1.0))
    smooth_signal = float(np.clip(smooth_surface_ratio / 0.85, 0.0, 1.0))
    blue_signal = float(np.clip(blue_ratio / 0.60, 0.0, 1.0))
    gray_signal = float(np.clip(gray_ratio / 0.35, 0.0, 1.0))
    muddy_signal = float(np.clip(muddy_ratio / 0.45, 0.0, 1.0))
    tone_signal = max(blue_signal, gray_signal, muddy_signal)

    image_score = 100.0 * (
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
        "image_damage_score": float(np.clip(image_score, 0.0, 100.0)),
        "analysis_confidence": round(float(np.clip(analysis_confidence, 0.0, 100.0)), 1),
        "summary": "Image-only flood analysis suggests that " + ", ".join(notes) + ".",
        "signals": {
            "Water extent": round(open_water_signal * 100, 1),
            "Muddy water": round(muddy_signal * 100, 1),
            "Gray floodwater": round(gray_signal * 100, 1),
            "Blue-water likelihood": round(blue_signal * 100, 1),
            "Surface smoothness": round(smooth_signal * 100, 1),
        },
        "overlay_caption": "Detected flood-water footprint",
        "overlay_color": (35, 181, 211),
        "coverage_label": "Water coverage",
        "score_label": "Flood damage score (image only)",
    }


def _storm_damage_analysis(rgb_image: np.ndarray) -> Dict[str, object]:
    """Detect storm-related building damage using debris and structural breakup cues."""
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

    fragmentation_seed = (
        (edge_density >= edge_high) & (gradient >= gradient_high)
    ) & ~vegetation & ~bright_cloud
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
            (
                debris_mask
                | exposed_ground_mask
                | bright_rubble_mask
                | shadow_damage_mask
            )
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
    image_score = 100.0 * building_damage_signal
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
        "image_damage_score": float(np.clip(image_score, 0.0, 100.0)),
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
        "score_label": "Storm damage score (image only)",
    }


def season_from_month(month: int) -> str:
    """Use Asia-relevant seasonal groupings for context features."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Pre-monsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def _make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _scaled_log_signal(values: pd.Series) -> pd.Series:
    """Convert heavy-tailed impact values into stable 0-1 historical signals."""
    series = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    transformed = np.log1p(series)
    positive = transformed[transformed > 0]
    upper = float(positive.quantile(0.99)) if not positive.empty else 1.0
    upper = max(upper, 1.0)
    return np.clip(transformed / upper, 0, 1)


def _build_duration_days(dataframe: pd.DataFrame) -> pd.Series:
    """Derive event duration from EM-DAT date components with safe fallbacks."""
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
    """Fill missing coordinates using country medians and static centroids."""
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

    seed = pd.util.hash_pandas_object(
        resolved["disno"].astype(str), index=False
    ).astype("uint64")
    lat_jitter = ((seed % 11).astype(float) - 5.0) * 0.18
    lon_jitter = (((seed // 11) % 11).astype(float) - 5.0) * 0.22

    has_map = resolved["map_latitude"].notna() & resolved["map_longitude"].notna()
    resolved.loc[has_map, "map_latitude"] = (
        resolved.loc[has_map, "map_latitude"] + lat_jitter[has_map]
    )
    resolved.loc[has_map, "map_longitude"] = (
        resolved.loc[has_map, "map_longitude"] + lon_jitter[has_map]
    )
    return resolved


def _build_detection_overlay(
    rgb_image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]
) -> Image.Image:
    """Highlight detected flood water or storm damage on the uploaded image."""
    base = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    highlighted = base.copy()
    if mask.any():
        color_layer = np.zeros_like(base)
        color_layer[:, :] = np.array(color, dtype=np.uint8)
        highlighted[mask] = (
            0.35 * base[mask] + 0.65 * color_layer[mask]
        ).astype(np.uint8)
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
    """
    Load EM-DAT records and prepare the Asia flood/storm analytics dataset.

    The final severity target is derived from historical deaths, affected
    population, and damage only, which keeps the training grounded in past data.
    """
    raw = pd.read_csv(csv_path)

    disno_fallback = pd.Series(raw.index.astype(str), index=raw.index)

    prepared = pd.DataFrame(
        {
            "disno": raw["DisNo."].fillna(disno_fallback).astype(str),
            "disaster_type": raw["Disaster Type"].fillna("Unknown").astype(str),
            "disaster_subtype": raw["Disaster Subtype"]
            .fillna("Unspecified")
            .astype(str),
            "country": raw["Country"].fillna("Unknown").astype(str),
            "subregion": raw["Subregion"].fillna("Unknown").astype(str),
            "region": raw["Region"].fillna("Unknown").astype(str),
            "event_name": raw["Event Name"].fillna("").astype(str).str.strip(),
            "location": raw["Location"].fillna("Location not reported").astype(str),
            "start_year": pd.to_numeric(raw["Start Year"], errors="coerce"),
            "start_month": pd.to_numeric(raw["Start Month"], errors="coerce")
            .fillna(1),
            "start_day": pd.to_numeric(raw["Start Day"], errors="coerce").fillna(1),
            "end_year": pd.to_numeric(raw["End Year"], errors="coerce"),
            "end_month": pd.to_numeric(raw["End Month"], errors="coerce"),
            "end_day": pd.to_numeric(raw["End Day"], errors="coerce"),
            "magnitude": pd.to_numeric(raw["Magnitude"], errors="coerce"),
            "latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "total_deaths": pd.to_numeric(raw["Total Deaths"], errors="coerce")
            .fillna(0),
            "total_affected": pd.to_numeric(raw["Total Affected"], errors="coerce")
            .fillna(0),
            "damage_total_k": pd.to_numeric(
                raw["Total Damage ('000 US$)"], errors="coerce"
            ),
            "damage_adjusted_k": pd.to_numeric(
                raw["Total Damage, Adjusted ('000 US$)"], errors="coerce"
            ),
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

    prepared["severity_score"] = 100.0 * (
        0.30 * deaths_signal + 0.50 * affected_signal + 0.20 * damage_signal
    )
    prepared["severity_band"] = prepared["severity_score"].apply(severity_band_label)
    prepared["total_affected_log"] = np.log1p(
        prepared["total_affected"].clip(lower=0)
    )

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
    prepared["country_mean_affected_log"] = np.log1p(
        prepared["country_mean_affected"].fillna(0).clip(lower=0)
    )

    prepared = _resolve_map_coordinates(prepared)
    prepared = prepared.sort_values(
        ["start_year", "severity_score"], ascending=[False, False]
    ).reset_index(drop=True)
    return prepared


class DisasterSeverityPredictor:
    """Train and serve the historical linear, random-forest, and image hybrid model."""

    numeric_features = [
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
            self.historical_df.groupby("country")["subregion"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict()
        )
        self.country_type_baseline = (
            self.historical_df.groupby(["country", "disaster_type"])["severity_score"]
            .mean()
            .to_dict()
        )
        self.global_defaults = {
            "country_event_count": float(
                self.historical_df["country_event_count"].median()
            ),
            "country_mean_severity": float(
                self.historical_df["country_mean_severity"].median()
            ),
            "country_mean_affected_log": float(
                self.historical_df["country_mean_affected_log"].median()
            ),
            "country_mean_damage": float(
                self.historical_df["country_mean_damage"].median()
            ),
            "magnitude": float(
                self.historical_df.loc[
                    self.historical_df["magnitude"].notna(), "magnitude"
                ].median()
            ),
        }

        self.linear_pipeline: Optional[Pipeline] = None
        self.random_forest_pipeline: Optional[Pipeline] = None
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance: pd.DataFrame = pd.DataFrame()
        self.training_frame = self._build_training_frame()
        self.train()

    def _build_preprocessor(self) -> ColumnTransformer:
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
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

    def _build_training_frame(self) -> pd.DataFrame:
        frame = self.historical_df.copy()
        frame["reference_year"] = frame["start_year"].astype(int)
        frame["disaster_subtype"] = (
            frame["disaster_subtype"].replace("", "Unspecified").fillna("Unspecified")
        )
        frame["magnitude"] = pd.to_numeric(frame["magnitude"], errors="coerce")
        return frame

    def _evaluate_model(
        self, model_name: str, predictions: np.ndarray, targets: pd.Series
    ) -> None:
        clipped_predictions = np.clip(predictions, 0, 100)
        self.model_metrics[model_name] = {
            "r2": float(r2_score(targets, clipped_predictions)),
            "mae": float(mean_absolute_error(targets, clipped_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, clipped_predictions))),
        }

    def _run_kfold_cv(self, pipeline_template, features: pd.DataFrame, target: pd.Series, model_name: str, n_splits: int = 5) -> Dict[str, float]:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        r2_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kfold.split(features):
            x_train_fold = features.iloc[train_idx]
            y_train_fold = target.iloc[train_idx]
            x_val_fold = features.iloc[val_idx]
            y_val_fold = target.iloc[val_idx]
            
            fold_pipeline = clone(pipeline_template)
            fold_pipeline.fit(x_train_fold, y_train_fold)
            predictions = fold_pipeline.predict(x_val_fold)
            clipped_predictions = np.clip(predictions, 0, 100)
            
            r2_scores.append(r2_score(y_val_fold, clipped_predictions))
            mae_scores.append(mean_absolute_error(y_val_fold, clipped_predictions))
        
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
        for feature_name, importance in zip(
            transformed_feature_names, model.feature_importances_
        ):
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
            grouped_importance[label] = grouped_importance.get(label, 0.0) + float(
                importance
            )

        importance_frame = pd.DataFrame(
            [{"feature": key, "importance": value} for key, value in grouped_importance.items()]
        ).sort_values("importance", ascending=False)
        return importance_frame.reset_index(drop=True)

    def train(self) -> None:
        """Train holdout-evaluated linear and random-forest severity models."""
        features = self.training_frame[self.numeric_features + self.categorical_features]
        target = self.training_frame["severity_score"]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=42,
        )

        linear_template = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor()),
                ("model", LinearRegression()),
            ]
        )
        rf_template = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=320,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        linear_holdout = clone(linear_template)
        linear_holdout.fit(x_train, y_train)
        self._evaluate_model(
            "Linear Regression", linear_holdout.predict(x_test), y_test
        )

        rf_holdout = clone(rf_template)
        rf_holdout.fit(x_train, y_train)
        self._evaluate_model("Random Forest", rf_holdout.predict(x_test), y_test)

        self.cv_results = {}
        linear_cv = self._run_kfold_cv(linear_template, features, target, "Linear Regression")
        self.cv_results["Linear Regression"] = linear_cv
        self.model_metrics["Linear Regression"]["cv_r2"] = linear_cv["cv_r2_mean"]
        self.model_metrics["Linear Regression"]["cv_mae"] = linear_cv["cv_mae_mean"]
        
        rf_cv = self._run_kfold_cv(rf_template, features, target, "Random Forest")
        self.cv_results["Random Forest"] = rf_cv
        self.model_metrics["Random Forest"]["cv_r2"] = rf_cv["cv_r2_mean"]
        self.model_metrics["Random Forest"]["cv_mae"] = rf_cv["cv_mae_mean"]

        self.linear_pipeline = linear_template.fit(features, target)
        self.random_forest_pipeline = rf_template.fit(features, target)
        self.feature_importance = self._build_feature_importance()

    def _country_stats_for_prediction(self, country: str) -> Dict[str, float]:
        subset = self.historical_df[self.historical_df["country"] == country]
        if subset.empty:
            return self.global_defaults.copy()
        return {
            "country_event_count": float(subset["country_event_count"].median()),
            "country_mean_severity": float(subset["country_mean_severity"].median()),
            "country_mean_affected_log": float(
                subset["country_mean_affected_log"].median()
            ),
            "country_mean_damage": float(subset["country_mean_damage"].median()),
            "magnitude": float(subset["magnitude"].dropna().median())
            if subset["magnitude"].notna().any()
            else 0.0,
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
        """Convert user inputs into the trained feature schema."""
        subregion = self.country_to_subregion.get(country, "Unknown")
        country_stats = self._country_stats_for_prediction(country)
        resolved_magnitude = (
            country_stats["magnitude"] if magnitude in (None, 0) else float(magnitude)
        )

        scenario = pd.DataFrame(
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
                    "country_mean_affected_log": country_stats[
                        "country_mean_affected_log"
                    ],
                    "country_mean_damage": country_stats["country_mean_damage"],
                }
            ]
        )
        return scenario

    def analyze_satellite_image(
        self, image_bytes: bytes, disaster_type: str
    ) -> Dict[str, object]:
        """
        Estimate a damage score from a single uploaded satellite image.

        This is intentionally described as an image analyzer rather than a
        supervised image model because the repository does not include labeled
        flood/storm imagery for training.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        image.thumbnail((512, 512))

        rgb = np.asarray(image).astype(np.float32) / 255.0
        analysis = (
            _flood_water_analysis(rgb)
            if disaster_type == "Flood"
            else _storm_damage_analysis(rgb)
        )
        detection_mask = analysis["mask"]
        detection_overlay = _build_detection_overlay(
            rgb,
            detection_mask,
            analysis["overlay_color"],
        )
        
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
            "image_size": {
                "width": int(original_size[0]),
                "height": int(original_size[1]),
            },
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
        """Return the closest historical Asia incidents for analyst context."""
        candidates = self.historical_df[
            self.historical_df["disaster_type"] == disaster_type
        ].copy()
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
            candidates["subregion"]
            == self.country_to_subregion.get(country, "Unknown"),
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
        """Predict severity by blending linear, random-forest, and image signals."""
        if self.linear_pipeline is None or self.random_forest_pipeline is None:
            raise ValueError("Models are not trained.")

        scenario_frame = self.build_scenario_frame(
            reference_year=reference_year,
            disaster_type=disaster_type,
            country=country,
            disaster_subtype=disaster_subtype,
            start_month=start_month,
            duration_days=duration_days,
            total_affected=total_affected,
            magnitude=magnitude,
        )

        linear_score = float(
            np.clip(self.linear_pipeline.predict(scenario_frame)[0], 0, 100)
        )
        rf_score = float(
            np.clip(self.random_forest_pipeline.predict(scenario_frame)[0], 0, 100)
        )

        image_result = None
        if image_bytes:
            image_result = self.analyze_satellite_image(image_bytes, disaster_type)
        fusion_decision = decide_image_fusion(rf_score, image_result)

        if image_result and fusion_decision["use_image"]:
            weights = {
                "Linear Regression": 0.20,
                "Random Forest": 0.55,
                "Satellite Image": 0.25,
            }
            hybrid_score = (
                weights["Linear Regression"] * linear_score
                + weights["Random Forest"] * rf_score
                + weights["Satellite Image"]
                * float(image_result["image_damage_score"])
            )
            all_scores = [
                linear_score,
                rf_score,
                float(image_result["image_damage_score"]),
            ]
        else:
            weights = {"Linear Regression": 0.25, "Random Forest": 0.75}
            if image_result:
                weights["Satellite Image"] = 0.0
            hybrid_score = (
                weights["Linear Regression"] * linear_score
                + weights["Random Forest"] * rf_score
            )
            all_scores = [linear_score, rf_score]

        model_spread = float(np.std(all_scores))
        hybrid_mae = (
            weights["Linear Regression"]
            * self.model_metrics["Linear Regression"]["mae"]
            + weights["Random Forest"] * self.model_metrics["Random Forest"]["mae"]
        )
        if image_result and fusion_decision["use_image"]:
            hybrid_mae += weights["Satellite Image"] * 9.0

        interval_radius = min(22.0, hybrid_mae + model_spread * 0.65)
        lower_bound = float(np.clip(hybrid_score - interval_radius, 0, 100))
        upper_bound = float(np.clip(hybrid_score + interval_radius, 0, 100))

        confidence_base = 88.0 if fusion_decision["use_image"] else 82.0
        confidence = float(
            np.clip(confidence_base - model_spread * 0.7, 58.0, 96.0)
        )

        baseline_key = (country, disaster_type)
        type_mask = self.historical_df["disaster_type"] == disaster_type
        baseline = float(
            self.country_type_baseline.get(
                baseline_key,
                self.historical_df.loc[type_mask, "severity_score"].mean(),
            )
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
        country_mean_affected = np.expm1(
            self._country_stats_for_prediction(country)["country_mean_affected_log"]
        )
        if total_affected > country_mean_affected * 1.5:
            drivers.append(
                "affected population is above the country's historical norm"
            )
        if duration_days >= 14:
            drivers.append("long event duration increases severity")
        if hybrid_score >= baseline + 8:
            drivers.append("the scenario sits above the country's historical baseline")
        if image_result and fusion_decision["use_image"]:
            drivers.append("satellite evidence agrees with the model and raises the final score")
        elif image_result:
            drivers.append("satellite evidence was reviewed separately but excluded from fusion")
        if not drivers:
            drivers.append("historical patterns keep the scenario near baseline")

        return {
            "hybrid_score": round(float(np.clip(hybrid_score, 0, 100)), 1),
            "prediction_mode": (
                "Model + Satellite Fusion"
                if fusion_decision["use_image"]
                else "Model Only"
            ),
            "severity_band": severity_band_label(float(hybrid_score)),
            "confidence": round(confidence, 1),
            "prediction_range": {
                "lower": round(lower_bound, 1),
                "upper": round(upper_bound, 1),
            },
            "component_scores": {
                "Linear Regression": round(linear_score, 1),
                "Random Forest": round(rf_score, 1),
            },
            "component_weights": weights,
            "image_only_damage_score": (
                round(float(image_result["image_damage_score"]), 1)
                if image_result
                else None
            ),
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
            self.historical_df.loc[
                self.historical_df["disaster_type"] == disaster_type,
                "disaster_subtype",
            ]
            .fillna("Unspecified")
            .replace("", "Unspecified")
            .value_counts()
            .index.tolist()
        )
        return subtypes[:12] if subtypes else ["Unspecified"]

    def model_quality_table(self) -> pd.DataFrame:
        """Return holdout metrics in a display-friendly frame."""
        rows = []
        for model_name, metrics in self.model_metrics.items():
            row = {
                "Model": model_name,
                "R2": round(metrics["r2"], 3),
                "MAE": round(metrics["mae"], 2),
                "RMSE": round(metrics["rmse"], 2),
            }
            if "cv_r2" in metrics:
                row["CV R2 (5-fold)"] = f"{metrics['cv_r2']:.3f} ± {self.cv_results.get(model_name, {}).get('cv_r2_std', 0):.3f}"
            if "cv_mae" in metrics:
                row["CV MAE (5-fold)"] = f"{metrics['cv_mae']:.2f} ± {self.cv_results.get(model_name, {}).get('cv_mae_std', 0):.2f}"
            rows.append(row)
        return pd.DataFrame(rows)


def decide_image_fusion(
    model_score: float, image_result: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """Blend image analysis only when the visual score is stable and aligned."""
    if not image_result:
        return {
            "use_image": False,
            "fusion_ratio": 0.0,
            "agreement_ratio": 0.0,
            "reason": "No satellite image was uploaded, so the final prediction uses only the trained models.",
        }

    image_score = float(image_result["image_damage_score"])
    image_confidence = float(image_result.get("analysis_confidence", 0.0))
    mask_coverage = float(image_result.get("mask_coverage", 0.0)) / 100.0
    score_gap = abs(model_score - image_score)

    agreement_ratio = float(np.clip(1.0 - (score_gap / 35.0), 0.0, 1.0))
    confidence_ratio = float(np.clip(image_confidence / 100.0, 0.0, 1.0))
    coverage_ratio = float(np.clip(mask_coverage / 0.12, 0.0, 1.0))
    fusion_ratio = float(
        np.clip(
            0.55 * agreement_ratio + 0.30 * confidence_ratio + 0.15 * coverage_ratio,
            0.0,
            1.0,
        )
    )
    use_image = bool(
        fusion_ratio >= 0.58 and image_confidence >= 52.0 and score_gap <= 24.0
    )

    if use_image:
        reason = (
            "The satellite score is close to the model estimate and the detected flood or damage pattern is strong enough to blend into the final severity score."
        )
    elif image_confidence < 52.0:
        reason = (
            "The uploaded image does not show strong enough flood-water or building-damage evidence, so the final prediction stays model-only."
        )
    else:
        reason = (
            "The satellite score is too far from the random-forest estimate, so the final prediction stays model-only."
        )

    return {
        "use_image": use_image,
        "fusion_ratio": round(fusion_ratio, 3),
        "agreement_ratio": round(agreement_ratio, 3),
        "score_gap": round(score_gap, 1),
        "image_confidence": round(image_confidence, 1),
        "reason": reason,
    }


@st.cache_data(show_spinner=False)
def get_historical_data(csv_path: str) -> pd.DataFrame:
    """Cache the prepared Asia disaster history used by all pages."""
    return load_and_prepare_disaster_data(csv_path)


@st.cache_resource(show_spinner=False)
def get_predictor(csv_path: str) -> DisasterSeverityPredictor:
    """Cache the trained predictor so the app does not retrain on every rerun."""
    return DisasterSeverityPredictor(load_and_prepare_disaster_data(csv_path))


@st.cache_data(show_spinner=False)
def get_legacy_model_summary(model_path: str) -> Dict[str, str]:
    """Expose the notebook-saved linear model as a legacy reference artifact."""
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
    """Apply dashboard filters while preserving the Asia 2000-2025 scope."""
    filtered = dataframe[
        dataframe["start_year"].between(year_range[0], year_range[1], inclusive="both")
    ].copy()
    if disaster_types:
        filtered = filtered[filtered["disaster_type"].isin(disaster_types)]
    if country != "All":
        filtered = filtered[filtered["country"] == country]
    if subregions:
        filtered = filtered[filtered["subregion"].isin(subregions)]
    return filtered.reset_index(drop=True)


def build_yearly_events_chart(filtered_df: pd.DataFrame):
    yearly = (
        filtered_df.groupby(["start_year", "disaster_type"])
        .agg(events=("disno", "size"))
        .reset_index()
    )
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
    return fig


def build_severity_distribution_chart(filtered_df: pd.DataFrame):
    fig = px.histogram(
        filtered_df,
        x="severity_score",
        color="disaster_type",
        nbins=20,
        opacity=0.80,
        color_discrete_map=DISASTER_COLORS,
        labels={"severity_score": "Severity score", "count": "Events"},
        title="Severity distribution",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return fig


def build_country_summary_chart(filtered_df: pd.DataFrame):
    country_summary = (
        filtered_df.groupby("country")
        .agg(
            events=("disno", "size"),
            avg_severity=("severity_score", "mean"),
            affected=("total_affected", "sum"),
        )
        .sort_values(["events", "avg_severity"], ascending=[False, False])
        .head(12)
        .reset_index()
    )
    fig = px.bar(
        country_summary,
        x="country",
        y="events",
        color="avg_severity",
        color_continuous_scale="YlOrRd",
        hover_data={"affected": ":,", "avg_severity": ":.1f"},
        labels={"country": "Country", "events": "Events", "avg_severity": "Avg severity"},
        title="Top countries in the filtered view",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def build_feature_importance_chart(feature_importance_df: pd.DataFrame, top_n: int = 8):
    if feature_importance_df.empty:
        fig = px.bar(
            x=[0], y=["No data"], text=["No data"],
            labels={"x": "Feature Importance", "y": ""},
            title="Feature Importance (Why this prediction?)"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        return fig
    
    top_features = feature_importance_df.head(top_n).copy()
    top_features = top_features.sort_values("importance", ascending=True)
    
    fig = px.bar(
        top_features,
        y="feature",
        x="importance",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
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
    return fig


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
    # Keep only Asia-view coordinates to avoid stray/outlier points on the Africa side.
    # (Plotly "asia" scope is view-only; we still need to filter points explicitly.)
    map_ready = map_ready[
        map_ready["map_latitude"].between(-15, 65, inclusive="both")
        & map_ready["map_longitude"].between(30, 155, inclusive="both")
    ].copy()
    fig = px.scatter_geo(
        map_ready,
        lat="map_latitude",
        lon="map_longitude",
        color="disaster_type",
        size="severity_score",
        size_max=16,
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
        lataxis_range=[-15, 65],
        lonaxis_range=[30, 155],
    )
    fig.update_layout(
        height=720,
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
                "Blend weight (%)": round(
                    float(prediction["component_weights"].get(name, 0.0)) * 100, 1
                ),
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
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        yaxis_range=[0, 100],
    )
    return fig


def render_past_data_page(dataframe: pd.DataFrame) -> None:
    st.title("Past Disaster Data")
    st.caption(
        "Historical EM-DAT events for Asia only, limited to floods and storms from January 1, 2000 through December 31, 2025."
    )

    filter_columns = st.columns(3)
    disaster_type_label = filter_columns[0].selectbox(
        "Disaster type", ["All", "Flood", "Storm"], index=0
    )
    year_range = filter_columns[1].slider(
        "Year range", min_value=2000, max_value=2025, value=(2000, 2025)
    )
    country = filter_columns[2].selectbox(
        "Country", ["All"] + sorted(dataframe["country"].unique().tolist())
    )

    selected_types = (
        ["Flood", "Storm"] if disaster_type_label == "All" else [disaster_type_label]
    )
    filtered = filter_historical_data(
        dataframe, disaster_types=selected_types, year_range=year_range, country=country
    )

    if filtered.empty:
        st.warning("No events match the selected filters.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Events", f"{len(filtered):,}")
    metric_columns[1].metric("Countries", f"{filtered['country'].nunique():,}")
    metric_columns[2].metric(
        "People affected", format_compact_number(filtered["total_affected"].sum())
    )
    metric_columns[3].metric(
        "Average severity", f"{filtered['severity_score'].mean():.1f}/100"
    )

    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(
        build_yearly_events_chart(filtered), use_container_width=True
    )
    chart_columns[1].plotly_chart(
        build_severity_distribution_chart(filtered), use_container_width=True
    )

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


def render_map_page(dataframe: pd.DataFrame) -> None:
    st.title("Asia Disaster Map")
    st.caption(
        "A world-view map focused on Asia so you can inspect where flood and storm events cluster across subregions."
    )

    filter_columns = st.columns(3)
    disaster_types = filter_columns[0].multiselect(
        "Disaster types", ["Flood", "Storm"], default=["Flood", "Storm"]
    )
    year_range = filter_columns[1].slider(
        "Year range", min_value=2000, max_value=2025, value=(2000, 2025), key="map_years"
    )
    all_subregions = sorted(dataframe["subregion"].dropna().unique().tolist())
    selected_subregions = filter_columns[2].multiselect(
        "Subregions", all_subregions, default=all_subregions
    )

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

    country_fig = px.bar(
        country_summary,
        x="country",
        y="events",
        color="avg_severity",
        color_continuous_scale="YlOrRd",
        labels={"country": "Country", "events": "Events", "avg_severity": "Avg severity"},
        title="Most active countries",
    )
    country_fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False
    )
    summary_columns[0].plotly_chart(country_fig, use_container_width=True)

    subregion_fig = px.bar(
        subregion_summary,
        x="subregion",
        y="events",
        color="avg_severity",
        color_continuous_scale="YlGnBu",
        labels={"subregion": "Subregion", "events": "Events", "avg_severity": "Avg severity"},
        title="Subregion activity",
    )
    subregion_fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False
    )
    summary_columns[1].plotly_chart(subregion_fig, use_container_width=True)


def render_prediction_page(
    predictor: DisasterSeverityPredictor,
    legacy_model_summary: Dict[str, str],
) -> None:
    st.title("Severity Prediction and Satellite Analysis")
    st.caption(
        "The model is trained on Asia flood and storm events from January 1, 2000 through December 31, 2025. Years above 2025 are forward-looking scenario inputs."
    )

    quality_table = predictor.model_quality_table().sort_values(
        ["R2", "MAE"], ascending=[False, True]
    )
    if not quality_table.empty:
        top_model = quality_table.iloc[0]["Model"]
        st.info(
            f"Best holdout model in this app: `{top_model}`. Random Forest is used as the primary upgraded model."
        )
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

    if legacy_model_summary.get("available"):
        st.caption(
            f"Legacy notebook artifact detected in `mlr_model.pkl`: `{legacy_model_summary['estimator']}`. The app keeps that linear approach as a baseline and upgrades the final prediction with a random forest."
        )
    else:
        st.caption(
            legacy_model_summary.get("message", "Legacy notebook artifact unavailable.")
        )

    default_country = (
        "India"
        if "India" in predictor.available_countries
        else predictor.available_countries[0]
    )
    disaster_type = st.selectbox("Disaster type", ["Flood", "Storm"], key="pred_type")
    subtype_options = predictor.available_subtypes(disaster_type)

    input_columns = st.columns(3)
    reference_year = input_columns[0].slider(
        "Reference year", min_value=2000, max_value=2030, value=2026
    )
    country = input_columns[1].selectbox(
        "Country",
        predictor.available_countries,
        index=predictor.available_countries.index(default_country),
    )
    disaster_subtype = input_columns[2].selectbox("Subtype", subtype_options, index=0)

    scenario_columns = st.columns(4)
    start_month = scenario_columns[0].slider("Start month", 1, 12, 7)
    duration_days = scenario_columns[1].number_input(
        "Duration (days)", min_value=1, max_value=180, value=10, step=1
    )
    total_affected = scenario_columns[2].number_input(
        "Estimated affected population",
        min_value=0,
        max_value=100_000_000,
        value=50_000,
        step=1_000,
    )
    magnitude = scenario_columns[3].number_input(
        "Magnitude (0 = auto from history)",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.1,
    )

    uploaded_file = st.file_uploader(
        "Upload a flood or storm satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Flood images are checked for water extent. Storm images are checked for building-damage signatures.",
    )

    if st.button("Run prediction", type="primary"):
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
        with st.spinner("Scoring the scenario and analyzing the satellite image..."):
            st.session_state["prediction_result"] = predictor.predict(
                reference_year=reference_year,
                disaster_type=disaster_type,
                country=country,
                disaster_subtype=disaster_subtype,
                start_month=start_month,
                duration_days=int(duration_days),
                total_affected=int(total_affected),
                magnitude=None if magnitude == 0 else float(magnitude),
                image_bytes=uploaded_bytes,
            )
            st.session_state["prediction_image"] = uploaded_bytes

    result = st.session_state.get("prediction_result")
    if not result:
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Predicted severity", f"{result['hybrid_score']}/100")
    metric_columns[1].metric("Severity band", result["severity_band"])
    metric_columns[2].metric("Confidence", f"{result['confidence']}%")
    metric_columns[3].metric(
        "Prediction range",
        f"{result['prediction_range']['lower']} - {result['prediction_range']['upper']}",
    )

    st.progress(
        min(max(float(result["hybrid_score"]) / 100.0, 0.0), 1.0),
        text=f"{result['prediction_mode']} | Baseline difference: {result['difference_from_baseline']:+.1f}",
    )

    if result["image_analysis"]:
        st.subheader("Image-only damage analysis")
        image_metric_columns = st.columns(4)
        image_metric_columns[0].metric(
            result["image_analysis"]["score_label"],
            f"{result['image_only_damage_score']}/100",
        )
        image_metric_columns[1].metric(
            result["image_analysis"]["coverage_label"],
            f"{result['image_analysis']['mask_coverage']}%",
        )
        image_metric_columns[2].metric(
            "Image confidence",
            f"{result['image_analysis']['analysis_confidence']}%",
        )
        image_metric_columns[3].metric(
            "Used in final severity",
            "Yes" if result["fusion_decision"]["use_image"] else "No",
        )
        st.caption(result["image_analysis"]["summary"])

        fusion_columns = st.columns(2)
        fusion_columns[0].metric(
            "Fusion ratio",
            f"{float(result['fusion_decision']['fusion_ratio']) * 100:.0f}%",
        )
        fusion_columns[1].metric(
            "Model/image gap", f"{result['fusion_decision']['score_gap']}"
        )
    st.caption(result["fusion_decision"]["reason"])

    st.plotly_chart(build_component_chart(result), use_container_width=True)

    st.subheader("Main drivers")
    st.markdown("\n".join(f"- {driver}" for driver in result["drivers"]))

    if result["image_analysis"]:
        image_columns = st.columns(2)
        uploaded_bytes = st.session_state.get("prediction_image")
        if uploaded_bytes:
            image_columns[0].image(
                uploaded_bytes,
                caption="Uploaded satellite image",
                use_container_width=True,
            )
        image_columns[1].image(
            result["image_analysis"]["detection_overlay"],
            caption=result["image_analysis"]["overlay_caption"],
            use_container_width=True,
        )

        signal_frame = (
            pd.DataFrame(
                [
                    {"Signal": key, "Strength (%)": value}
                    for key, value in result["image_analysis"]["signals"].items()
                ]
            )
            .sort_values("Strength (%)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(signal_frame, use_container_width=True, hide_index=True)

    similar_events = result["similar_events"].copy()
    if not similar_events.empty:
        if "Severity Score" in similar_events.columns:
            similar_events["Severity Score"] = similar_events["Severity Score"].round(1)
        if "Damage (M US$)" in similar_events.columns:
            similar_events["Damage (M US$)"] = similar_events["Damage (M US$)"].round(2)
        st.subheader("Most similar historical events")
        st.dataframe(similar_events, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Asia Disaster Intelligence",
        page_icon=":earth_asia:",
        layout="wide",
    )

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    historical_df = get_historical_data(str(DATA_PATH))
    predictor = get_predictor(str(DATA_PATH))
    legacy_model_summary = get_legacy_model_summary(str(LEGACY_MODEL_PATH))

    st.sidebar.title("Asia Disaster Intelligence")
    st.sidebar.caption("Flood and storm analytics for Asia, 2000-2025.")
    page = st.sidebar.radio(
        "Open page",
        ["Past Data", "Asia Map", "Prediction"],
    )

    st.sidebar.metric("Records", f"{len(historical_df):,}")
    st.sidebar.metric("Countries", f"{historical_df['country'].nunique():,}")
    st.sidebar.metric(
        "Average severity", f"{historical_df['severity_score'].mean():.1f}/100"
    )
    if legacy_model_summary.get("available"):
        st.sidebar.caption(
            f"Legacy notebook model detected: {legacy_model_summary['estimator']}"
        )

    if page == "Past Data":
        render_past_data_page(historical_df)
    elif page == "Asia Map":
        render_map_page(historical_df)
    else:
        render_prediction_page(predictor, legacy_model_summary)


if __name__ == "__main__":
    main()
