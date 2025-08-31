# Anomaly Detection Project (UCL)
---

## 📌 Overview

Generative AI technologies can be maliciously used for misinformation, making detection tools crucial to keep up with the rapid evolution of synthetic content. This project frames the detection of fake audio as an **anomaly detection problem**. By leveraging **Stochastic Differential Equation (SDE) modeling** and **interpretable machine learning**, the goal is to develop a robust pipeline for identifying synthetic audio in time series data.

---

## Introduction

The proliferation of generative audio models introduces challenges in verifying the authenticity of spoken content. Detecting synthetic audio is critical for:

- Preventing misinformation
- Ensuring biometric security
- Maintaining the integrity of digital media

By analyzing **Mel spectrogram time series** and augmenting them with statistical and machine learning classifiers, we construct an interpretable and effective detection framework.

---

## Stochastic Differential Equations (SDEs)

SDEs describe systems influenced by both deterministic trends and inherent randomness. A general SDE has the form:

dXₜ = μ(Xₜ, t) dt + σ(Xₜ, t) dWₜ

Where:  

- Xₜ is the stochastic process  
- μ(Xₜ, t) is the drift term (deterministic trend)  
- σ(Xₜ, t) is the diffusion term (random fluctuation)  
- Wₜ is a Wiener process (standard Brownian motion)  

SDE realizations are single sample paths generated numerically (e.g., using Euler–Maruyama). Each realization represents a possible evolution of the process over time.

---

## Audio Processing Pipeline

### Preprocessing

- Audio waveforms normalized and converted to **Mel spectrograms**  
- Captures spectral and temporal dynamics  
- Statistical features extracted per Mel band (e.g., range, smoothness, increment statistics)  

### SDE Fitting

- Each Mel band’s time series modeled with an SDE  
- Drift and noise residuals estimated for anomaly detection  

### Global Spectral Features

- MFCCs  
- Spectral centroid, bandwidth, flatness, roll-off  
- Zero-crossing rate  

### Feature Representation

- Combines deterministic trends with stochastic variations for anomaly detection  

---

## Models

### 1. Non-ML Rule-Based Model

- Computes **Z-scores** for each feature relative to real audio corpus  
- Flags features exceeding threshold as abnormal  
- Classifies clip as fake if the number of abnormal features surpasses a second threshold  
- Provides interpretable anomaly detection based solely on statistical principles  

### 2. Random Forest Classifier

- Ensemble of decision trees trained on random subsets of features  
- Robust to overfitting and effective for high-dimensional data  
- Provides feature importance scores for interpretability  
- SHAP values explain which features contribute most to predictions  

---

## Dataset Construction

- **Real Audio:** 12 audiobook chapters, split into ~30s chunks  
- **Fake Audio:** 36 clips across chapters 1, 4, and 5  
- Split into:  
  - Corpus (70% real)  
  - Validation (15% real + 50% fake)  
  - Test (15% real + 50% fake)  
- Stratified sampling maintains balance of real vs fake clips  

---

## Feature Extraction

- Extracted 40+ features including:  
  - MFCCs  
  - Spectral centroid mean/std  
  - Zero-crossing rate  
  - SDE residuals  
- Constructed data frames for **corpus**, **validation**, and **test sets**  

---

## Performance Evaluation

- Confusion matrices used to evaluate model performance  
- Non-ML algorithm had only **1 false positive**, showing strong performance with limited data  
- SHAP analysis used to interpret ML model predictions  

---

## Applications

This anomaly detection framework can be applied in:

- Finance (fraud detection)  
- Healthcare (signal anomaly detection)  
- Cybersecurity  
- Manufacturing & IoT  
- Aerospace & Transportation  
- Audio, speech, and multimedia authentication  
- Social media verification  
- Environmental monitoring  

---

## Project Timeline

| Week | Activity |
|------|----------|
| 1    | Stochastic Differential Equations (SDEs) |
| 2    | Anomaly Detection for SDEs |
| 3    | Detecting Fake Audios & Presentation |
