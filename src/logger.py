"""
src/logger.py
=============
Session and event logging for ergonomic posture monitoring.

Provides structured logging of:
  1. Periodic posture detections (telemetry stream at ~1 Hz or on state change).
  2. Posture alert trigger events (when continuous bad posture exceeds threshold).
  3. Session summary analytics (percentage breakdown of good vs. bad postures).

Logs are stored in `logs/` as standard CSV files and accessible directly in-memory
for Streamlit download and dashboard display.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DETECTIONS_LOG_PATH = LOGS_DIR / "posture_detections.csv"
ALERTS_LOG_PATH = LOGS_DIR / "posture_alerts.csv"


class PostureLogger:
    """Manages session-level posture detection and alert logging."""

    DETECTION_HEADERS = [
        "timestamp",
        "session_id",
        "posture_label",
        "confidence_pct",
        "neck_flexion_deg",
        "trunk_inclination_deg",
        "shoulder_asymmetry_deg",
        "alert_active",
    ]

    ALERT_HEADERS = [
        "timestamp",
        "session_id",
        "posture_label",
        "confidence_pct",
        "duration_exceeded_s",
        "neck_flexion_deg",
        "trunk_inclination_deg",
        "shoulder_asymmetry_deg",
        "alert_type",
    ]

    def __init__(self, session_id: str | None = None):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"SESSION_{now_str}"

        # Initialize files with headers if they don't exist
        self._ensure_file(DETECTIONS_LOG_PATH, self.DETECTION_HEADERS)
        self._ensure_file(ALERTS_LOG_PATH, self.ALERT_HEADERS)

        # In-memory history for live session visualization
        self.session_detections: list[dict[str, Any]] = []
        self.session_alerts: list[dict[str, Any]] = []
        self._last_log_time: float = 0.0

    @staticmethod
    def _ensure_file(path: Path, headers: list[str]) -> None:
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_detection(
        self,
        posture_label: str,
        confidence_pct: float,
        angles: dict[str, float] | None,
        alert_active: bool,
        current_time: float,
        interval_s: float = 0.5,
    ) -> bool:
        """
        Record a periodic posture detection. Throttled by *interval_s*
        unless an alert state is active, to keep log files compact.
        """
        if current_time - self._last_log_time < interval_s:
            return False

        self._last_log_time = current_time
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        record = {
            "timestamp": ts,
            "session_id": self.session_id,
            "posture_label": posture_label,
            "confidence_pct": round(confidence_pct, 2),
            "neck_flexion_deg": round(angles.get("neck_flexion_deg", 0.0), 1) if angles else 0.0,
            "trunk_inclination_deg": round(angles.get("trunk_inclination_deg", 0.0), 1) if angles else 0.0,
            "shoulder_asymmetry_deg": round(angles.get("shoulder_asymmetry_deg", 0.0), 1) if angles else 0.0,
            "alert_active": alert_active,
        }

        self.session_detections.append(record)

        with open(DETECTIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.DETECTION_HEADERS)
            writer.writerow(record)

        return True

    def log_alert(
        self,
        posture_label: str,
        confidence_pct: float,
        duration_exceeded_s: float,
        angles: dict[str, float] | None,
        alert_type: str = "Audio Beep + Visual Flash",
    ) -> dict[str, Any]:
        """Record an alert trigger event when continuous bad posture threshold is reached."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        record = {
            "timestamp": ts,
            "session_id": self.session_id,
            "posture_label": posture_label,
            "confidence_pct": round(confidence_pct, 2),
            "duration_exceeded_s": round(duration_exceeded_s, 2),
            "neck_flexion_deg": round(angles.get("neck_flexion_deg", 0.0), 1) if angles else 0.0,
            "trunk_inclination_deg": round(angles.get("trunk_inclination_deg", 0.0), 1) if angles else 0.0,
            "shoulder_asymmetry_deg": round(angles.get("shoulder_asymmetry_deg", 0.0), 1) if angles else 0.0,
            "alert_type": alert_type,
        }

        self.session_alerts.append(record)

        with open(ALERTS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.ALERT_HEADERS)
            writer.writerow(record)

        return record

    def get_session_stats(self) -> dict[str, Any]:
        """Compute aggregated posture statistics for the current monitoring session."""
        if not self.session_detections:
            return {
                "total_samples": 0,
                "good_posture_pct": 0.0,
                "bad_posture_pct": 0.0,
                "total_alerts": len(self.session_alerts),
                "distribution": {},
            }

        df = pd.DataFrame(self.session_detections)
        # Exclude "No Pose Detected" from ergonomic distribution
        valid_df = df[df["posture_label"] != "No Pose Detected"]
        total = len(valid_df)

        if total == 0:
            return {
                "total_samples": len(df),
                "good_posture_pct": 0.0,
                "bad_posture_pct": 0.0,
                "total_alerts": len(self.session_alerts),
                "distribution": {},
            }

        counts = valid_df["posture_label"].value_counts().to_dict()
        good_count = counts.get("Good Posture", 0)
        good_pct = (good_count / total) * 100.0
        bad_pct = 100.0 - good_pct

        distribution = {cls: round((cnt / total) * 100.0, 1) for cls, cnt in counts.items()}

        return {
            "total_samples": total,
            "good_posture_pct": round(good_pct, 1),
            "bad_posture_pct": round(bad_pct, 1),
            "total_alerts": len(self.session_alerts),
            "distribution": distribution,
        }

    def get_detections_df(self) -> pd.DataFrame:
        """Return the current session's detections as a pandas DataFrame."""
        if not self.session_detections:
            return pd.DataFrame(columns=self.DETECTION_HEADERS)
        return pd.DataFrame(self.session_detections)

    def get_alerts_df(self) -> pd.DataFrame:
        """Return the current session's alerts as a pandas DataFrame."""
        if not self.session_alerts:
            return pd.DataFrame(columns=self.ALERT_HEADERS)
        return pd.DataFrame(self.session_alerts)
