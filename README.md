# AI-Based Ergonomic Posture Monitoring for Computer Users

An undergraduate Final Year Project (SAIA3122 FYP 1) implementing real-time computer vision and machine learning for ergonomic posture assessment using a standard web camera.

---

## 📌 Project Overview

Prolonged computer use in non-ergonomic positions is a primary contributor to Work-Related Musculoskeletal Disorders (WMSDs). This application provides non-intrusive, real-time posture monitoring and early intervention alerts without requiring wearable sensors or saving private video recordings to disk.

### Target Performance Metrics
| Metric | Specification Target | Achieved |
|---|---|---|
| **Classification Accuracy** | > 90.0% | **96.00%** |
| **Throughput / Latency** | $\ge$ 30 FPS | Capable of real-time 30+ FPS stream |
| **Alert Latency** | Configurable threshold | Default: **3.0 s** continuous poor posture |
| **Privacy / Security** | In-memory frame processing only | No video or images stored on disk |

---

## 🎯 Target Posture Classes

1. **Good Posture** – Upright spine, neck aligned with shoulder girdle ($\approx 90^\circ$), horizontal shoulder alignment.
2. **Slouching (Trunk Inclination)** – Forward lean of the torso between hip midpoint and shoulder midpoint.
3. **Forward Head Posture (Neck Flexion)** – Protrusion of the head relative to the shoulder girdle.
4. **Asymmetric Tilt (Shoulder Slope)** – Lateral tilt or uneven elevation of left vs. right shoulders.

---

## 📐 Biomechanical Feature Engineering

Using **MediaPipe Pose** (normalized landmark coordinates), the pipeline extracts key anatomical landmarks:
- **Left/Right Ears**: Indices `7` and `8`
- **Left/Right Shoulders**: Indices `11` and `12`
- **Left/Right Hips**: Indices `23` and `24`

### Key Angle Computations (`math.atan2`):
- **Neck Flexion ($^\circ$)**: Angle of the vector connecting shoulder midpoint to ear midpoint relative to the vertical axis.
- **Trunk Inclination ($^\circ$)**: Angle of the vector connecting hip midpoint to shoulder midpoint relative to the vertical axis.
- **Shoulder Asymmetry ($^\circ$)**: Signed horizontal slope angle across left and right shoulder landmarks.

---

## 📂 Project Structure

```text
AI-Based Ergonomic Posture Monitoring for Computer Users/
│
├── .gitignore                    # Python, venv, Streamlit cache, *.pkl exclusions
├── requirements.txt              # Core project dependencies
├── README.md                     # Documentation and execution instructions
├── posture_rf_model.pkl          # Serialized Random Forest classifier + class labels
│
├── data/
│   └── synthetic_posture_data.csv# Generated dataset (8,000 samples, 2,000/class)
│
├── src/
│   ├── __init__.py               # Package marker
│   ├── features.py               # MediaPipe landmark extraction & angle math
│   ├── utils.py                  # In-memory WAV synthesis (base64) & OpenCV HUD overlay
│   └── train_model.py            # Synthetic dataset generator, training & evaluation
│
└── app.py                        # Streamlit web dashboard with real-time video stream
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Requirements
- **Python**: 3.12 (64-bit recommended)
- **OS**: Windows 10/11
- **Hardware**: Standard USB or integrated webcam

### 2. Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Train Model & Generate Dataset
```powershell
python src/train_model.py
```
Outputs:
- `data/synthetic_posture_data.csv`
- `posture_rf_model.pkl`

### 5. Launch Live Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`, adjust webcam settings on the left sidebar, and click **▶ Start Monitoring**.

---

## 📊 Evaluation Results (Test Set)

```text
==================================================
  Samples - train: 6400  test: 1600
  Accuracy   : 0.9600  (96.00 %)
  Precision  : 0.9602
  Recall     : 0.9600
  F1-Score   : 0.9599

  Confusion Matrix:
  Classes: ['Asymmetric Tilt', 'Forward Head Posture', 'Good Posture', 'Slouching']
  [[381    0   14    5]
   [  0  389    2    9]
   [  3    1  394    2]
   [  6   17    5  372]]
==================================================
```
