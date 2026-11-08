"""
src/features.py
===============
MediaPipe landmark extraction and omnidirectional biomechanical feature engineering
supporting BOTH front-facing (facing camera) and profile (side view) desk setups.

Landmarks used (MediaPipe Pose indices):
  0  – Nose (fallback anchor)
  7  – Left ear
  8  – Right ear
 11  – Left shoulder
 12  – Right shoulder
 23  – Left hip
 24  – Right hip

Biomechanical 3D Angle Calculations:
1. Neck Flexion (deg)       – Head alignment relative to shoulder girdle in 3D.
                              Frontal/Side forward lean measured via sqrt(dx^2 + dz^2).
                              Upright ≈ 85°-90°; Forward tilt drops to 60°-75°.
2. Trunk Inclination (deg)  – Torso inclination in 3D (sagittal plane + optical depth).
                              Upright ≈ 85°-90°; Slouching drops to 65°-78°.
3. Shoulder Asymmetry (deg) – Elevation disparity between left and right shoulders.
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


def _midpoint_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the 3D midpoint (x, y, z) between two coordinate tuples."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _angle_deg_3d(
    vertical_span: float,
    horizontal_span_x: float,
    depth_span_z: float,
    depth_weight: float = 1.35,
) -> float:
    """
    Calculate angle in degrees relative to the horizontal plane (0° = horizontal, 90° = vertical upright).

    Supports omnidirectional posture detection:
      - horizontal_span_x: captures forward lean in side / profile view.
      - depth_span_z: captures forward lean along camera depth in front-facing view.
      - Total horizontal displacement = sqrt(dx^2 + (depth_weight * dz)^2).
    """
    horizontal_disp = math.sqrt(horizontal_span_x ** 2 + (depth_span_z * depth_weight) ** 2)
    return math.degrees(math.atan2(abs(vertical_span), horizontal_disp + 1e-6))


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_features(pose_landmarks: Any) -> dict[str, float] | None:
    """
    Extract biomechanical features from detected MediaPipe pose landmarks.

    Omnidirectional design:
      - Accurately captures slouching and forward head posture whether the user
        is facing directly toward the webcam or sitting in side view.
      - Gracefully handles desk occlusion using anatomical priors and depth estimation.

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

    ls_z = float(getattr(ls, "z", 0.0) or 0.0)
    rs_z = float(getattr(rs, "z", 0.0) or 0.0)

    left_shoulder  = (float(ls.x), float(ls.y), ls_z)
    right_shoulder = (float(rs.x), float(rs.y), rs_z)
    shoulder_mid   = _midpoint_3d(left_shoulder, right_shoulder)

    # ── 2. Head / Ears (With nose fallback if ears occluded) ─────────────────
    le = lm[LM_LEFT_EAR] if len(lm) > LM_LEFT_EAR else None
    re = lm[LM_RIGHT_EAR] if len(lm) > LM_RIGHT_EAR else None

    vis_le = getattr(le, "visibility", 1.0) if le else 0.0
    vis_re = getattr(re, "visibility", 1.0) if re else 0.0
    vis_le = 1.0 if vis_le is None else vis_le
    vis_re = 1.0 if vis_re is None else vis_re

    le_z = float(getattr(le, "z", shoulder_mid[2]) or shoulder_mid[2]) if le else shoulder_mid[2]
    re_z = float(getattr(re, "z", shoulder_mid[2]) or shoulder_mid[2]) if re else shoulder_mid[2]

    if le and re and vis_le > 0.25 and vis_re > 0.25:
        left_ear  = (float(le.x), float(le.y), le_z)
        right_ear = (float(re.x), float(re.y), re_z)
        ear_mid   = _midpoint_3d(left_ear, right_ear)
    elif le and vis_le > 0.25:
        left_ear  = (float(le.x), float(le.y), le_z)
        right_ear = (left_ear[0] + 0.15, left_ear[1], le_z)
        ear_mid   = left_ear
    elif re and vis_re > 0.25:
        right_ear = (float(re.x), float(re.y), re_z)
        left_ear  = (right_ear[0] - 0.15, right_ear[1], re_z)
        ear_mid   = right_ear
    else:
        # Fallback to nose / face center if ears are covered by headphones
        nose = lm[LM_NOSE]
        nose_z = float(getattr(nose, "z", shoulder_mid[2]) or shoulder_mid[2])
        ear_mid   = (float(nose.x), float(nose.y), nose_z)
        left_ear  = (float(nose.x) - 0.08, float(nose.y), nose_z)
        right_ear = (float(nose.x) + 0.08, float(nose.y), nose_z)

    # ── 3. Hips (Seated estimation) ──────────────────────────────────────────
    # At a computer desk, hips are often below the frame edge or behind the desk.
    # MediaPipe estimates 3D depth and position; if off-screen, infer using anatomical prior.
    has_hips = len(lm) > LM_RIGHT_HIP
    if has_hips:
        lh = lm[LM_LEFT_HIP]
        rh = lm[LM_RIGHT_HIP]
        lh_z = float(getattr(lh, "z", 0.0) or 0.0)
        rh_z = float(getattr(rh, "z", 0.0) or 0.0)

        # Use MediaPipe's coordinates if they are reasonably valid
        if 0.0 < lh.y < 1.4 and 0.0 < rh.y < 1.4:
            left_hip  = (float(lh.x), float(lh.y), lh_z)
            right_hip = (float(rh.x), float(rh.y), rh_z)
            hip_mid   = _midpoint_3d(left_hip, right_hip)
        else:
            # Anatomical prior: hips are ~0.35 below shoulders in normalized coords
            # Preserve MediaPipe predicted depth for frontal slouching detection
            estimated_hip_z = (lh_z + rh_z) / 2.0
            hip_mid   = (shoulder_mid[0], min(0.95, shoulder_mid[1] + 0.35), estimated_hip_z)
            left_hip  = (shoulder_mid[0] - 0.10, hip_mid[1], lh_z)
            right_hip = (shoulder_mid[0] + 0.10, hip_mid[1], rh_z)
    else:
        hip_mid   = (shoulder_mid[0], min(0.95, shoulder_mid[1] + 0.35), shoulder_mid[2] + 0.05)
        left_hip  = (shoulder_mid[0] - 0.10, hip_mid[1], hip_mid[2])
        right_hip = (shoulder_mid[0] + 0.10, hip_mid[1], hip_mid[2])

    # ── 4. Omnidirectional Biomechanical Angle Calculations ─────────────────

    # 1. Neck Flexion (3D):
    #    Vector from shoulder midpoint up to head/ear midpoint.
    #    Vertical span: dy = shoulder_mid[y] - ear_mid[y]
    #    Lateral span: dx = shoulder_mid[x] - ear_mid[x] (side view lean)
    #    Depth span: dz = shoulder_mid[z] - ear_mid[z] (frontal forward head jut)
    #    Upright = ~88°-90°; Forward Head = drops to 60°-75°
    neck_flexion_deg = _angle_deg_3d(
        vertical_span=shoulder_mid[1] - ear_mid[1],
        horizontal_span_x=shoulder_mid[0] - ear_mid[0],
        depth_span_z=shoulder_mid[2] - ear_mid[2],
        depth_weight=1.4,
    )

    # 2. Trunk Inclination (3D):
    #    Vector from hip midpoint up to shoulder midpoint.
    #    Vertical span: dy = hip_mid[y] - shoulder_mid[y]
    #    Lateral span: dx = hip_mid[x] - shoulder_mid[x] (side view slouch)
    #    Depth span: dz = hip_mid[z] - shoulder_mid[z] (frontal forward slouch towards screen)
    #    Upright = ~88°-90°; Torso Slouch = drops to 65°-78°
    trunk_inclination_deg = _angle_deg_3d(
        vertical_span=hip_mid[1] - shoulder_mid[1],
        horizontal_span_x=hip_mid[0] - shoulder_mid[0],
        depth_span_z=hip_mid[2] - shoulder_mid[2],
        depth_weight=1.5,
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
