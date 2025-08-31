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

A **Stochastic Differential Equation (SDE)** is like an ordinary differential equation (ODE) 
but with an additional noise term that models stochastic (random) fluctuations.

The computational definition of an SDE with Gaussian noise is:

X(t + Δt) = X(t) + f(X, t) Δt + g(X, t) √(Δt) ξ

where ξ is a normally distributed random variable with zero mean and unit variance. 
The value of X(t + Δt) is computed from X(t), and this process is repeated for each time step.

Formally, an SDE can be written as:

dXₜ = f(Xₜ, t) dt + g(Xₜ, t) dWₜ

- f(Xₜ, t) — drift term (deterministic trend)  
- g(Xₜ, t) — diffusion term (magnitude of stochastic effects)  
- Wₜ — Wiener process (Brownian motion)  
- dWₜ — infinitesimal random increment with mean 0 and variance dt


