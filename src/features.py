"""
src/features.py
===============
MediaPipe landmark extraction and biomechanical feature engineering
tailored for seated computer desk users.

Landmarks used (MediaPipe Pose indices):
  7  – Left ear
  8  – Right ear
 11  – Left shoulder
 12  – Right shoulder
 23  – Left hip
 24  – Right hip

Derived biomechanical angles:
1. Neck Flexion (deg)       – Head alignment relative to shoulder girdle.
                              Upright ≈ 85°-90°; Forward tilt drops to 60°-75°.
2. Trunk Inclination (deg)  – Torso alignment relative to horizontal plane.
                              Upright ≈ 85°-90°; Slouching drops to 65°-78°.
3. Shoulder Asymmetry (deg) – Elevation angle across the left & right shoulders.
                              Level ≈ 0°-2°; Asymmetric tilt reaches 8°-15°+.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ── Landmark index constants ─────────────────────────────────────────────────
LM_NOSE           = 0
LM_LEFT_EAR       = 7
LM_RIGHT_EAR      = 8
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24

# ── Feature column names (must match model training order) ───────────────────
FEATURE_COLUMNS = [
    "neck_flexion_deg",
    "trunk_inclination_deg",
    "shoulder_asymmetry_deg",
    # Raw normalised coordinates for auxiliary context
    "left_ear_x",    "left_ear_y",
    "right_ear_x",   "right_ear_y",
    "left_shoulder_x",  "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_hip_x",    "left_hip_y",
    "right_hip_x",   "right_hip_y",
]


def _angle_deg(vertical_span: float, horizontal_span: float) -> float:
    """
    Calculate angle in degrees relative to the horizontal axis (0 deg = horizontal, 90 deg = vertical).
    For upright posture, vertical span >> horizontal span, yielding ~85°-90°.
    """
    return math.degrees(math.atan2(abs(vertical_span), abs(horizontal_span) + 1e-6))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Return the midpoint (x, y) between two coordinate tuples."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_features(pose_landmarks: Any) -> dict[str, float] | None:
    """
    Extract biomechanical features from detected MediaPipe pose landmarks.

    Designed for seated computer users where hips might be partially occluded
    by a desk. Gracefully infers occluded landmarks instead of rejecting frames.

    Parameters
    ----------
    pose_landmarks : list or object with .landmark property
        Normalized landmarks from MediaPipe Pose.

    Returns
    -------
    dict[str, float] or None
        Feature dictionary keyed by FEATURE_COLUMNS, or None only if
        no human upper body / shoulders are detected in the frame.
    """
    if pose_landmarks is None:
        return None

    # Handle both legacy object (`pose_landmarks.landmark`) and Tasks API (direct list)
    if hasattr(pose_landmarks, "landmark"):
        lm = pose_landmarks.landmark
    else:
        lm = pose_landmarks

    if len(lm) < 13:  # Must at least have shoulders
        return None

    # ── 1. Shoulders (Primary anchor for upper-body presence) ────────────────
    ls = lm[LM_LEFT_SHOULDER]
    rs = lm[LM_RIGHT_SHOULDER]

    vis_ls = getattr(ls, "visibility", 1.0)
    vis_rs = getattr(rs, "visibility", 1.0)
    vis_ls = 1.0 if vis_ls is None else vis_ls
    vis_rs = 1.0 if vis_rs is None else vis_rs

    # If neither shoulder is detected with reasonable presence, no person is in frame
    if vis_ls < 0.2 and vis_rs < 0.2:
        return None

    left_shoulder  = (ls.x, ls.y)
    right_shoulder = (rs.x, rs.y)
    shoulder_mid   = _midpoint(left_shoulder, right_shoulder)

    # ── 2. Head / Ears (With nose fallback if ears occluded) ─────────────────
    le = lm[LM_LEFT_EAR] if len(lm) > LM_LEFT_EAR else None
    re = lm[LM_RIGHT_EAR] if len(lm) > LM_RIGHT_EAR else None

    vis_le = getattr(le, "visibility", 1.0) if le else 0.0
    vis_re = getattr(re, "visibility", 1.0) if re else 0.0
    vis_le = 1.0 if vis_le is None else vis_le
    vis_re = 1.0 if vis_re is None else vis_re

    if le and re and vis_le > 0.25 and vis_re > 0.25:
        left_ear  = (le.x, le.y)
        right_ear = (re.x, re.y)
        ear_mid   = _midpoint(left_ear, right_ear)
    elif le and vis_le > 0.25:
        left_ear  = (le.x, le.y)
        right_ear = (left_ear[0] + 0.15, left_ear[1])
        ear_mid   = left_ear
    elif re and vis_re > 0.25:
        right_ear = (re.x, re.y)
        left_ear  = (right_ear[0] - 0.15, right_ear[1])
        ear_mid   = right_ear
    else:
        # Fallback to nose / face center if ears are covered by headphones
        nose = lm[LM_NOSE]
        ear_mid   = (nose.x, nose.y)
        left_ear  = (nose.x - 0.08, nose.y)
        right_ear = (nose.x + 0.08, nose.y)

    # ── 3. Hips (Seated estimation) ──────────────────────────────────────────
    # At a computer desk, hips are often below the frame edge or behind the desk.
    # MediaPipe estimates their position; if off-screen, infer using anatomical prior.
    has_hips = len(lm) > LM_RIGHT_HIP
    if has_hips:
        lh = lm[LM_LEFT_HIP]
        rh = lm[LM_RIGHT_HIP]
        # Use MediaPipe's coordinates if they are reasonably valid
        if 0.0 < lh.y < 1.4 and 0.0 < rh.y < 1.4:
            left_hip  = (lh.x, lh.y)
            right_hip = (rh.x, rh.y)
            hip_mid   = _midpoint(left_hip, right_hip)
        else:
            # Anatomical prior: hips are ~0.35 below shoulders in normalized coords
            hip_mid   = (shoulder_mid[0], min(0.95, shoulder_mid[1] + 0.35))
            left_hip  = (shoulder_mid[0] - 0.10, hip_mid[1])
            right_hip = (shoulder_mid[0] + 0.10, hip_mid[1])
    else:
        hip_mid   = (shoulder_mid[0], min(0.95, shoulder_mid[1] + 0.35))
        left_hip  = (shoulder_mid[0] - 0.10, hip_mid[1])
        right_hip = (shoulder_mid[0] + 0.10, hip_mid[1])

    # ── 4. Biomechanical Angle Calculations ─────────────────────────────────

    # 1. Neck Flexion:
    #    Angle of the vector from shoulder midpoint up to head/ear midpoint.
    #    Vertical span: dy = shoulder_mid[y] - ear_mid[y] (positive when head is above shoulders)
    #    Horizontal span: dx = abs(shoulder_mid[x] - ear_mid[x])
    #    Upright = ~88°-90°; Head forward protrusion = drops to 60°-75°
    neck_flexion_deg = _angle_deg(
        shoulder_mid[1] - ear_mid[1],
        shoulder_mid[0] - ear_mid[0],
    )

    # 2. Trunk Inclination:
    #    Angle of the vector from hip midpoint up to shoulder midpoint.
    #    Vertical span: dy = hip_mid[y] - shoulder_mid[y]
    #    Horizontal span: dx = abs(hip_mid[x] - shoulder_mid[x])
    #    Upright = ~88°-90°; Torso forward slouch = drops to 65°-78°
    trunk_inclination_deg = _angle_deg(
        hip_mid[1] - shoulder_mid[1],
        hip_mid[0] - shoulder_mid[0],
    )

    # 3. Shoulder Asymmetry:
    #    Tilt angle of the shoulder line relative to horizontal.
    #    Vertical difference: dy = abs(right_shoulder[y] - left_shoulder[y])
    #    Horizontal span: dx = abs(right_shoulder[x] - left_shoulder[x])
    #    Level shoulders = ~0°-2°; One shoulder raised / tilted = 8°-15°+
    shoulder_asymmetry_deg = math.degrees(
        math.atan2(
            abs(right_shoulder[1] - left_shoulder[1]),
            abs(right_shoulder[0] - left_shoulder[0]) + 1e-6,
        )
    )

    return {
        "neck_flexion_deg":        float(neck_flexion_deg),
        "trunk_inclination_deg":   float(trunk_inclination_deg),
        "shoulder_asymmetry_deg":  float(shoulder_asymmetry_deg),
        # Raw coordinates for model inference
        "left_ear_x":        float(left_ear[0]),
        "left_ear_y":        float(left_ear[1]),
        "right_ear_x":       float(right_ear[0]),
        "right_ear_y":       float(right_ear[1]),
        "left_shoulder_x":   float(left_shoulder[0]),
        "left_shoulder_y":   float(left_shoulder[1]),
        "right_shoulder_x":  float(right_shoulder[0]),
        "right_shoulder_y":  float(right_shoulder[1]),
        "left_hip_x":        float(left_hip[0]),
        "left_hip_y":        float(left_hip[1]),
        "right_hip_x":       float(right_hip[0]),
        "right_hip_y":       float(right_hip[1]),
    }


def feature_dict_to_df(feature_dict: dict[str, float]) -> pd.DataFrame:
    """
    Convert a feature dictionary to a single-row DataFrame whose columns
    are ordered exactly as FEATURE_COLUMNS.
    """
    row = {col: feature_dict.get(col, np.nan) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
