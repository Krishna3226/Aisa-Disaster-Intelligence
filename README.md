# Asia Disaster Intelligence

A machine learning web dashboard for predicting flood and storm 
severity across Asia on a 0–10 scale.

## Features
- Historical EM-DAT disaster data explorer (2000–2025)
- Interactive Asia map with severity visualization
- Severity prediction using Linear Regression + Random Forest
- Satellite image analysis using OpenCV (flood/storm detection)
- Confidence-weighted bidirectional score fusion
- Temporal holdout validation (train 2000–2020, test 2021–2025)
- Intelligent model caching system

## Tech Stack
- Python 3.10+
- Streamlit
- scikit-learn
- Pandas, NumPy
- Plotly
- OpenCV
- Pillow
- Joblib

## Setup
pip install streamlit pandas numpy scikit-learn plotly pillow opencv-python joblib
streamlit run app.py

## Team
- [Your Name] — ML Models, Prediction Engine, Image Analysis
- [Member 2 Name] — Data Pipeline, Historical Analytics
- [Member 3 Name] — Frontend, System Architecture

## Dataset
EM-DAT International Disaster Database — filtered to Asia, 
floods and storms, 2000–2025.