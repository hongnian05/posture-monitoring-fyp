"""
src/utils.py
============
Utility helpers for the posture monitoring application.

Sections
--------
1. Audio Alert  – Generates a pure-tone WAV file entirely in memory
                  (stdlib only: wave, struct, math) and returns it as a
                  base64-encoded data-URI string suitable for Streamlit's
                  ``st.audio()`` or an HTML <audio> element.

2. HUD Overlay  – OpenCV drawing helpers that render landmark dots,
                  skeleton connections, angle readings, posture labels,
                  and FPS counter onto a BGR numpy frame.
"""

from __future__ import annotations

import base64
import io
import math
import struct
import wave
from typing import Any

import cv2
import numpy as np

# ── Posture class -> HUD colour (BGR) ─────────────────────────────────────────
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "Good Posture":          (0, 220, 0),    # green
    "Slouching":             (0, 100, 255),  # orange
    "Forward Head Posture":  (0, 50, 255),   # red-orange
    "Asymmetric Tilt":       (0, 0, 255),    # red
}
DEFAULT_COLOUR = (200, 200, 200)  # grey for unknown / no detection

# ── MediaPipe skeleton connections for the 6 landmarks we use ────────────────
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (7,  11),   # left ear  -> left shoulder
    (8,  12),   # right ear -> right shoulder
    (11, 12),   # left shoulder -> right shoulder
    (11, 23),   # left shoulder -> left hip
    (12, 24),   # right shoulder -> right hip
    (23, 24),   # left hip -> right hip
]

_LM_INDICES = [7, 8, 11, 12, 23, 24]


# =============================================================================
# 1. AUDIO ALERT
# =============================================================================

def _generate_sine_wav(
    frequency: float = 880.0,
    duration_s: float = 0.4,
    sample_rate: int = 44_100,
    amplitude: float = 0.6,
) -> bytes:
    """Synthesise a sine-wave tone and return raw PCM bytes of a valid WAV file."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate)
            envelope = 1.0 - (i / n_samples)
            packed = struct.pack("<h", int(sample * envelope * 32767))
            wf.writeframes(packed)

    return buf.getvalue()


def get_alert_audio_b64(
    frequency: float = 880.0,
    duration_s: float = 0.4,
) -> str:
    """Return a base64-encoded data:audio/wav;base64,... URI for the alert tone."""
    wav_bytes = _generate_sine_wav(frequency=frequency, duration_s=duration_s)
    encoded   = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


# =============================================================================
# 2. HUD OVERLAY
# =============================================================================

def _lm_to_px(lm: Any, w: int, h: int) -> tuple[int, int]:
    """Convert a normalised MediaPipe landmark to pixel coordinates safely clamped within frame."""
    x = int(max(0.0, min(1.0, lm.x)) * w)
    y = int(max(0.0, min(1.0, lm.y)) * h)
    return (x, y)


def draw_hud(
    frame: np.ndarray,
    pose_landmarks: Any | None,
    posture_label: str,
    angles: dict[str, float] | None,
    fps: float,
    alert_active: bool,
) -> np.ndarray:
    """
    Render the full HUD onto frame (modified in-place, also returned).

    Supports both:
      - MediaPipe legacy solutions: object with a `.landmark` property
      - MediaPipe modern Tasks API: direct list/sequence of landmark objects
    """
    h, w = frame.shape[:2]
    colour = CLASS_COLOURS.get(posture_label, DEFAULT_COLOUR)

    # ── Draw skeleton connections & landmarks ────────────────────────────────
    if pose_landmarks is not None:
        if hasattr(pose_landmarks, "landmark"):
            lm = pose_landmarks.landmark
        else:
            lm = pose_landmarks

        if len(lm) > max(_LM_INDICES):
            for idx_a, idx_b in SKELETON_CONNECTIONS:
                pt_a = _lm_to_px(lm[idx_a], w, h)
                pt_b = _lm_to_px(lm[idx_b], w, h)
                cv2.line(frame, pt_a, pt_b, colour, 2, cv2.LINE_AA)

            for idx in _LM_INDICES:
                cx, cy = _lm_to_px(lm[idx], w, h)
                cv2.circle(frame, (cx, cy), 5, colour, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Posture label banner ─────────────────────────────────────────────────
    label_text = f"Posture: {posture_label}"
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)

    # ── Angle readings panel (bottom-left) ──────────────────────────────────
    if angles is not None:
        panel_lines = [
            f"Neck Flex  : {angles['neck_flexion_deg']:+.1f} deg",
            f"Trunk Incl : {angles['trunk_inclination_deg']:+.1f} deg",
            f"Shoulder   : {angles['shoulder_asymmetry_deg']:+.1f} deg",
        ]
        panel_x, panel_y0 = 10, h - 20 - (len(panel_lines) * 22)
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x - 5, panel_y0 - 18),
                      (panel_x + 320, h - 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(panel_lines):
            cv2.putText(frame, line,
                        (panel_x, panel_y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)

    # ── FPS counter (top-right) ──────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, fps_text, (w - tw - 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    # ── Alert flash border ───────────────────────────────────────────────────
    if alert_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(frame, "! BAD POSTURE ALERT !",
                    (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    return frame
