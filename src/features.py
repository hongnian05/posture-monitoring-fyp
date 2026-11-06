"""
src/features.py
===============
MediaPipe landmark extraction and biomechanical feature engineering.

Landmarks used (MediaPipe Pose indices):
  7  – Left ear
  8  – Right ear
 11  – Left shoulder
 12  – Right shoulder
 23  – Left hip
 24  – Right hip

Derived biomechanical angles
----------------------------
1. Neck Flexion       – forward tilt of the head relative to the shoulder
                        girdle (Forward Head Posture indicator).
2. Trunk Inclination  – forward/backward lean of the torso between
                        shoulders and hips (Slouching indicator).
3. Shoulder Asymmetry – signed vertical height difference between left and
                        right shoulders (Asymmetric Tilt indicator).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ── Landmark index constants ─────────────────────────────────────────────────
LM_LEFT_EAR       = 7
LM_RIGHT_EAR      = 8
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24

# Ordered list for validation
REQUIRED_LANDMARKS = [
    LM_LEFT_EAR,
    LM_RIGHT_EAR,
    LM_LEFT_SHOULDER,
    LM_RIGHT_SHOULDER,
    LM_LEFT_HIP,
    LM_RIGHT_HIP,
]

# ── Feature column names (must match training order) ─────────────────────────
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


def _angle_deg(dy: float, dx: float) -> float:
    """Return the signed angle in degrees via arctan2(dy, dx)."""
    return math.degrees(math.atan2(dy, dx))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Return the midpoint between two (x, y) tuples."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_features(pose_landmarks: Any) -> dict[str, float] | None:
    """
    Extract biomechanical features from MediaPipe Pose landmarks.

    Supports both:
      - MediaPipe legacy solutions: object with a `.landmark` property
      - MediaPipe modern Tasks API: direct list/sequence of landmark objects

    Parameters
    ----------
    pose_landmarks : list or NormalizedLandmarkList
        The landmark list representing detected pose keypoints.

    Returns
    -------
    dict[str, float] or None
        Feature dictionary keyed by FEATURE_COLUMNS, or None when
        required landmarks are absent or have low visibility (< 0.5).
    """
    if pose_landmarks is None:
        return None

    # Handle both legacy object (`pose_landmarks.landmark`) and Tasks API (direct list)
    if hasattr(pose_landmarks, "landmark"):
        lm = pose_landmarks.landmark
    else:
        lm = pose_landmarks

    if len(lm) <= max(REQUIRED_LANDMARKS):
        return None

    # Validate that every required landmark is sufficiently visible
    for idx in REQUIRED_LANDMARKS:
        vis = getattr(lm[idx], "visibility", 1.0)
        if vis is not None and vis < 0.5:
            return None

    # ── Extract (x, y) in normalised image coordinates ─────────────────────
    left_ear       = (lm[LM_LEFT_EAR].x,       lm[LM_LEFT_EAR].y)
    right_ear      = (lm[LM_RIGHT_EAR].x,      lm[LM_RIGHT_EAR].y)
    left_shoulder  = (lm[LM_LEFT_SHOULDER].x,  lm[LM_LEFT_SHOULDER].y)
    right_shoulder = (lm[LM_RIGHT_SHOULDER].x, lm[LM_RIGHT_SHOULDER].y)
    left_hip       = (lm[LM_LEFT_HIP].x,       lm[LM_LEFT_HIP].y)
    right_hip      = (lm[LM_RIGHT_HIP].x,      lm[LM_RIGHT_HIP].y)

    # ── Derived midpoints ───────────────────────────────────────────────────
    ear_mid      = _midpoint(left_ear,      right_ear)       # head proxy
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)  # neck base
    hip_mid      = _midpoint(left_hip,      right_hip)       # pelvis proxy

    # ── Biomechanical angles ────────────────────────────────────────────────

    # 1. Neck Flexion
    neck_flexion_deg = _angle_deg(
        shoulder_mid[0] - ear_mid[0],   # horizontal offset
        shoulder_mid[1] - ear_mid[1],   # vertical offset
    )

    # 2. Trunk Inclination
    trunk_inclination_deg = _angle_deg(
        hip_mid[0] - shoulder_mid[0],   # horizontal offset
        hip_mid[1] - shoulder_mid[1],   # vertical offset
    )

    # 3. Shoulder Asymmetry
    shoulder_asymmetry_deg = _angle_deg(
        right_shoulder[1] - left_shoulder[1],   # height difference
        right_shoulder[0] - left_shoulder[0],   # horizontal span
    )

    return {
        "neck_flexion_deg":        neck_flexion_deg,
        "trunk_inclination_deg":   trunk_inclination_deg,
        "shoulder_asymmetry_deg":  shoulder_asymmetry_deg,
        # Raw coordinates for richer model context
        "left_ear_x":        left_ear[0],
        "left_ear_y":        left_ear[1],
        "right_ear_x":       right_ear[0],
        "right_ear_y":       right_ear[1],
        "left_shoulder_x":   left_shoulder[0],
        "left_shoulder_y":   left_shoulder[1],
        "right_shoulder_x":  right_shoulder[0],
        "right_shoulder_y":  right_shoulder[1],
        "left_hip_x":        left_hip[0],
        "left_hip_y":        left_hip[1],
        "right_hip_x":       right_hip[0],
        "right_hip_y":       right_hip[1],
    }


def feature_dict_to_df(feature_dict: dict[str, float]) -> pd.DataFrame:
    """
    Convert a feature dictionary to a single-row DataFrame whose columns
    are ordered exactly as FEATURE_COLUMNS.

    Parameters
    ----------
    feature_dict : dict[str, float]
        Output of extract_features.

    Returns
    -------
    pd.DataFrame
        Shape (1, len(FEATURE_COLUMNS)), ready for sklearn inference.
    """
    row = {col: feature_dict.get(col, np.nan) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
