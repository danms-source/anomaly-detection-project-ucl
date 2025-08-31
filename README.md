# Anomaly Detection Project (UCL)
---

## 📌 Overview

Generative AI technologies can be maliciously used for misinformation, making detection tools crucial to keep up with the rapid evolution of synthetic content. This project frames the detection of fake audio as an **anomaly detection problem**. By leveraging **Stochastic Differential Equation (SDE) modeling** and **interpretable machine learning**, the goal is to develop a robust pipeline for identifying synthetic audio in time series data.

---

## Project Timeline

| Week | Activity |
|------|----------|
| 1    | Stochastic Differential Equations (SDEs) |
| 2    | Anomaly Detection for SDEs |
| 3    | Detecting Fake Audios & Presentation |

---

## Week 1: Stochastic Differential Equations (SDEs)

### What are SDEs?  
A **Stochastic Differential Equation (SDE)** is an ordinary differential equation (ODE) with an additional noise term to model stochastic fluctuations.

The general form is:

<img width="433" height="75" alt="image" src="https://github.com/user-attachments/assets/ec1ab037-614d-44b0-aeb3-90773316086d" />

where:  

- f(Xₜ, t) — drift term (deterministic trend)  
- g(Xₜ, t) — diffusion term (magnitude of stochastic effects)  
- Wₜ — Wiener process (Brownian motion)
- dt — an infinitesimal time increment
- dWₜ — infinitesimal random increment with mean 0 and variance dt
