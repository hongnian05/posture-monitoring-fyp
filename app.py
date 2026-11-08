"""
app.py
======
Streamlit dashboard for real-time ergonomic posture monitoring.

Usage
-----
    streamlit run app.py

Features
--------
  • Live webcam feed via OpenCV (no video written to disk).
  • MediaPipe Pose landmark overlay (indices 7, 8, 11, 12, 23, 24).
  • Universal compatibility: supports both modern MediaPipe Tasks API
    and legacy MediaPipe Solutions API.
  • Real-time Random Forest classification of 4 posture classes.
  • FPS counter rendered on the HUD.
  • Visual (red border flash) + audio (sine-wave beep) alert when bad
    posture is detected continuously for >= ALERT_THRESHOLD_S seconds.
  • Persistent session detection & alert event logging (CSV exportable).
  • Sidebar controls: webcam index, alert threshold, confidence display.
"""

from __future__ import annotations

import base64
import importlib
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

# ── Project path resolution ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.features as features_mod
import src.logger as logger_mod
import src.utils as utils_mod

importlib.reload(features_mod)
importlib.reload(utils_mod)
importlib.reload(logger_mod)

from src.features import extract_features, feature_dict_to_df
from src.logger import PostureLogger
from src.utils import CLASS_COLOURS, draw_hud, get_alert_audio_b64

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH          = PROJECT_ROOT / "posture_rf_model.pkl"
POSE_TASK_PATH      = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
GOOD_POSTURE_LABEL  = "Good Posture"
DEFAULT_CAM_INDEX   = 0
DEFAULT_THRESHOLD_S = 3.0        # seconds of bad posture before alert fires
FRAME_BUFFER_LEN    = 10         # rolling window for FPS smoothing

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Ergonomic Posture Monitor",
    page_icon="🧍",
    layout="wide",
)


# ── Universal Pose Detector (Supports both Tasks API and legacy Solutions) ───
class UniversalPoseDetector:
    """Wrapper that abstracts over both MediaPipe Tasks API and legacy solutions."""

    def __init__(self, task_model_path: Path):
        self.is_tasks = False
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    smooth_landmarks=True,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
            else:
                raise AttributeError("Legacy solutions API not present")
        except Exception:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(task_model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.35,
                min_pose_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.is_tasks = True

    def process(self, rgb_frame: np.ndarray):
        """Process RGB image array and return normalized pose landmarks list."""
        if self.is_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                return result.pose_landmarks[0]
            return None
        else:
            result = self.detector.process(rgb_frame)
            return result.pose_landmarks


# ── Resource Caching ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> tuple | None:
    """Load the Random Forest artefact from disk."""
    if not MODEL_PATH.exists():
        return None
    artefact = joblib.load(MODEL_PATH)
    return artefact["model"], artefact["classes"]


@st.cache_resource
def load_pose_estimator() -> UniversalPoseDetector:
    """Initialise the universal MediaPipe Pose detector (cached singleton)."""
    return UniversalPoseDetector(POSE_TASK_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────────
def inject_audio_alert(audio_b64: str) -> None:
    """Inject an autoplay <audio> element into the page in-memory."""
    html = (
        f'<audio autoplay style="display:none">'
        f'<source src="{audio_b64}" type="audio/wav">'
        f'</audio>'
    )
    st.markdown(html, unsafe_allow_html=True)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open camera using DirectShow first (fast on Windows), then fallback."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


# ── Sidebar ──────────────────────────────────────────────────────────────────
def build_sidebar() -> dict:
    """Render the sidebar controls and return the user settings dict."""
    st.sidebar.title("⚙️ Controls & Settings")

    cam_index = st.sidebar.number_input(
        "Webcam Index", min_value=0, max_value=10,
        value=DEFAULT_CAM_INDEX, step=1,
        help="0 is the default integrated webcam; select 1 or 2 for external USB cameras.",
    )
    threshold_s = st.sidebar.slider(
        "Bad-Posture Alert Threshold (seconds)",
        min_value=1.0, max_value=10.0,
        value=DEFAULT_THRESHOLD_S, step=0.5,
        help="Alert fires when bad posture is maintained continuously for this long.",
    )
    show_confidence = st.sidebar.checkbox("Show Class Probabilities", value=True)
    enable_logging  = st.sidebar.checkbox("Enable Telemetry Logging", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Target Posture Classes")
    for cls, bgr in CLASS_COLOURS.items():
        r, g, b = bgr[2], bgr[1], bgr[0]
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        st.sidebar.markdown(
            f'<span style="color:{hex_col}; font-size:1.1rem; font-weight:bold;">■</span> **{cls}**',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Final Year Project (SAIA3122 FYP 1) | UTM")

    return {
        "cam_index":        int(cam_index),
        "threshold_s":      float(threshold_s),
        "show_confidence":  show_confidence,
        "enable_logging":   enable_logging,
    }


# ── Render Analytics Section ─────────────────────────────────────────────────
def render_analytics(logger: PostureLogger) -> None:
    """Render session analytics cards and downloadable CSV tables."""
    st.markdown("---")
    st.subheader("📋 Session Posture Analytics & Data Logs")

    stats = logger.get_session_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Good Posture Time", f"{stats['good_posture_pct']:.1f}%")
    m2.metric("Bad Posture Time", f"{stats['bad_posture_pct']:.1f}%")
    m3.metric("Total Alerts Fired", f"{stats['total_alerts']}")
    m4.metric("Logged Telemetry Samples", f"{stats['total_samples']}")

    col_l1, col_l2 = st.columns(2)
    alert_df = logger.get_alerts_df()
    det_df = logger.get_detections_df()

    with col_l1:
        st.markdown("#### 🚨 Alert Trigger Events Log")
        if not alert_df.empty:
            st.dataframe(alert_df.tail(20), use_container_width=True, height=220)
            st.download_button(
                "📥 Download Alert Logs (CSV)",
                data=alert_df.to_csv(index=False),
                file_name="posture_alerts_log.csv",
                mime="text/csv",
                key="dl_alerts",
            )
        else:
            st.info("No posture alerts triggered in this session yet.")

    with col_l2:
        st.markdown("#### 📈 Real-Time Telemetry Detections Log")
        if not det_df.empty:
            st.dataframe(det_df.tail(20), use_container_width=True, height=220)
            st.download_button(
                "📥 Download Detection Telemetry (CSV)",
                data=det_df.to_csv(index=False),
                file_name="posture_detections_log.csv",
                mime="text/csv",
                key="dl_detections",
            )
        else:
            st.info("No telemetry data recorded yet. Start monitoring to collect logs.")


# ── Main Application ─────────────────────────────────────────────────────────
def main() -> None:
    st.title("🧍 AI-Based Ergonomic Posture Monitor")
    st.markdown(
        "Real-time posture classification using **MediaPipe Pose** and **Random Forest (96.0% accuracy)**. "
        "Processed strictly in-memory with automatic session data logging."
    )

    settings = build_sidebar()

    # ── Initialize session logger ────────────────────────────────────────────
    if "posture_logger" not in st.session_state:
        st.session_state["posture_logger"] = PostureLogger()
    logger: PostureLogger = st.session_state["posture_logger"]

    # ── Load ML Model & Pose Detector ────────────────────────────────────────
    model_data = load_model()
    if model_data is None:
        st.error(
            f"⚠️ Trained model not found at `{MODEL_PATH}`.\n\n"
            "Run `python src/train_model.py` from the project root first."
        )
        return
    rf_model, classes = model_data

    try:
        pose_detector = load_pose_estimator()
    except Exception as e:
        st.error(f"❌ Failed to initialise pose detector: {e}")
        return

    # ── Layout: Video | Metrics ──────────────────────────────────────────────
    col_video, col_metrics = st.columns([3, 1])

    with col_video:
        st.subheader("📷 Live Camera Stream")
        monitor_active = st.toggle(
            "▶ Start Live Monitoring",
            value=False,
            help="Toggle to start or stop webcam analysis",
        )
        frame_placeholder = st.empty()

    with col_metrics:
        st.subheader("📊 Posture Status")
        label_placeholder = st.empty()
        fps_placeholder   = st.empty()
        alert_placeholder = st.empty()
        proba_placeholder = st.empty()

    # Initial standby UI when monitoring is inactive
    if not monitor_active:
        frame_placeholder.info(
            "👉 Toggle **'Start Live Monitoring'** above to turn on your webcam and start real-time posture analysis.\n\n"
            "**Optimal Setup Guidelines:**\n"
            "- Sit approximately 50-70 cm away from your webcam.\n"
            "- Keep your head, shoulders, and chest clearly visible in the frame.\n"
            "- Ensure adequate ambient lighting for landmark detection."
        )
        label_placeholder.metric("Status", "Standby")
        fps_placeholder.metric("FPS", "--")
        alert_placeholder.success("System Ready (No Alert)")

        # Render analytics history below
        render_analytics(logger)
        return

    # ── Pre-generate alert audio (in-memory) ─────────────────────────────────
    alert_audio_b64 = get_alert_audio_b64(frequency=880.0, duration_s=0.4)

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = open_camera(settings["cam_index"])
    if not cap.isOpened():
        frame_placeholder.error(
            f"❌ Unable to access webcam at index **{settings['cam_index']}**.\n\n"
            "- If you are using an external camera, change **Webcam Index** to `1` or `2` in the sidebar.\n"
            "- Ensure no other application (Zoom, Teams, Camera app) is currently using the camera."
        )
        render_analytics(logger)
        return

    # ── Frame loop state ─────────────────────────────────────────────────────
    frame_times: deque[float] = deque(maxlen=FRAME_BUFFER_LEN)
    bad_posture_start: float | None = None
    last_alert_time:   float        = 0.0
    ALERT_COOLDOWN_S                = 3.0
    posture_label                   = "Detecting..."
    angles_dict: dict | None        = None
    alert_active                    = False

    try:
        while monitor_active:
            ret, frame = cap.read()
            if not ret:
                frame_placeholder.warning("⚠️ Webcam frame dropped or camera disconnected.")
                break

            frame = cv2.flip(frame, 1)  # Mirror selfie view
            t_now = time.perf_counter()

            # ── MediaPipe inference ──────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose_detector.process(rgb_frame)

            # ── Feature extraction & classification ──────────────────────────
            probas: np.ndarray | None = None
            features = extract_features(landmarks)

            if features is not None:
                angles_dict   = {
                    "neck_flexion_deg":       features["neck_flexion_deg"],
                    "trunk_inclination_deg":  features["trunk_inclination_deg"],
                    "shoulder_asymmetry_deg": features["shoulder_asymmetry_deg"],
                }
                X_df          = feature_dict_to_df(features)
                posture_label = rf_model.predict(X_df)[0]
                probas        = rf_model.predict_proba(X_df)[0]
            else:
                posture_label = "No Pose Detected"
                angles_dict   = None

            # ── Bad-posture timer & alert logic ──────────────────────────────
            if posture_label != GOOD_POSTURE_LABEL and posture_label != "No Pose Detected":
                if bad_posture_start is None:
                    bad_posture_start = t_now
                elapsed = t_now - bad_posture_start
                alert_active = elapsed >= settings["threshold_s"]
            else:
                bad_posture_start = None
                alert_active      = False

            # ── Alert Trigger & Logging ──────────────────────────────────────
            confidence_pct = float(np.max(probas) * 100.0) if probas is not None else 0.0

            if alert_active and (t_now - last_alert_time) >= ALERT_COOLDOWN_S:
                inject_audio_alert(alert_audio_b64)
                last_alert_time = t_now
                if settings["enable_logging"]:
                    logger.log_alert(
                        posture_label=posture_label,
                        confidence_pct=confidence_pct,
                        duration_exceeded_s=elapsed,
                        angles=angles_dict,
                    )

            # Periodic telemetry log (~1 Hz)
            if settings["enable_logging"] and posture_label != "Detecting...":
                logger.log_detection(
                    posture_label=posture_label,
                    confidence_pct=confidence_pct,
                    angles=angles_dict,
                    alert_active=alert_active,
                    current_time=t_now,
                    interval_s=1.0,
                )

            # ── FPS calculation ──────────────────────────────────────────────
            frame_times.append(t_now)
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            else:
                fps = 0.0

            # ── HUD drawing ──────────────────────────────────────────────────
            annotated = draw_hud(
                frame,
                landmarks,
                posture_label,
                angles_dict,
                fps,
                alert_active,
            )

            # Convert BGR to RGB for Streamlit display (strictly in-memory)
            display_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # ── Metrics updates ──────────────────────────────────────────────
            cls_colour = CLASS_COLOURS.get(posture_label, (200, 200, 200))
            r, g, b    = cls_colour[2], cls_colour[1], cls_colour[0]
            hex_col    = f"#{r:02x}{g:02x}{b:02x}"
            label_placeholder.markdown(
                f"**Current Posture:**\n\n"
                f'<span style="color:{hex_col}; font-size:1.3rem; font-weight:bold;">'
                f'{posture_label}</span>',
                unsafe_allow_html=True,
            )
            fps_placeholder.metric("Inference Speed", f"{fps:.1f} FPS")

            if alert_active:
                alert_placeholder.error(
                    f"🚨 **ALERT!** Bad posture for >= {settings['threshold_s']:.1f}s!"
                )
            else:
                alert_placeholder.success("Posture Status: Normal")

            if settings["show_confidence"] and probas is not None:
                proba_md = "**Class Confidence:**\n"
                for cls_name, prob in sorted(zip(classes, probas), key=lambda t: -t[1]):
                    bar = "█" * int(prob * 18)
                    proba_md += f"\n`{cls_name[:18]:18s}` {prob*100:5.1f}% {bar}"
                proba_placeholder.markdown(proba_md)

    finally:
        cap.release()
        render_analytics(logger)


if __name__ == "__main__":
    main()
