# 🤟 Real-Time ASL Alphabet Recognition Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline-yellow.svg)

## 📌 Overview
An End-to-End Machine Learning Computer Vision project that detects and classifies American Sign Language (ASL) alphabets in real-time. Instead of relying on pre-built datasets, this project features a custom-built automated data collection script that extracts 21 hand landmarks using **MediaPipe**. The coordinates are processed and fed into a custom-trained **Random Forest** model utilizing a Scikit-Learn `Pipeline` to ensure zero data leakage and robust real-time inference.

## 🎥 Real-Time Demo
*(قم برفع صورة متحركة GIF لنفسك أثناء تشغيل الكاميرا واستبدل هذا السطر برابط الصورة)*
`![ASL Demo](link_to_your_gif_here)`

## 🛠️ Tech Stack & Tools
* **Computer Vision:** OpenCV, Google MediaPipe (Hand Tracking)
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, StandardScaler, Pipeline)
* **Data Processing:** Pandas, NumPy

## 🚀 Key Features & Engineering Highlights
* **Custom Data Collection:** An interactive OpenCV script that captures dynamic hand landmarks and autosaves them into a structured CSV format.
* **Robust ML Pipeline:** Unified `StandardScaler` and `RandomForest` into a single serialized `.pkl` pipeline. This eliminates training-serving skew and dimensional mismatch errors during live inference.
* **Optimized Real-Time Execution:** Lightweight processing loop maintaining high FPS during webcam inference.

## 📂 Project Structure
```text
├── data_collection.py       # Script to capture webcam frames and build the CSV dataset
├── train_model.ipynb        # Data splitting, Pipeline creation, and model training
├── real_time_prediction.py  # Live inference script using the trained pipeline
├── dataset.csv              # The custom collected hand landmark dataset
├── model.pkl                # The serialized Scikit-Learn Pipeline
└── requirements.txt         # Project dependencies
