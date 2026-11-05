# Brake Fade Early Warning — Gradient Boosting Classifier

> Three-class brake-fade risk classification from synthetic thermal and braking signals

## Overview

This project demonstrates a brake-fade risk-classification workflow using synthetic automotive sensor variables.

Classes:
- No Fade
- Mild Fade
- Severe Fade

Inputs include:
- front and rear rotor temperature;
- caliper temperature;
- pad wear;
- pedal force;
- deceleration;
- braking duration;
- vehicle speed;
- consecutive stops.

## Model

```text
GradientBoostingClassifier
n_estimators = 200
learning_rate = 0.08
max_depth = 4
subsample = 0.85
```

Evaluation uses stratified 5-fold cross-validation plus a held-out 20% test split.

## Data Scope

The dataset is **synthetic and generated in code** using a rule-based fade score with added noise. It is intended to demonstrate the end-to-end classification workflow and should not be interpreted as a validated brake-system safety model.

## Outputs

The repository exports:
- confusion matrix and one-vs-rest ROC curves;
- feature importance;
- thermal feature-space visualization;
- class probability over a simulated track-style temperature progression;
- CV accuracy and macro-F1 in `metrics.json`.

## Run

```bash
git clone https://github.com/bharath153/Brake_Fade_GradientBoosting.git
cd Brake_Fade_GradientBoosting
pip install -r requirements.txt
python src/train_classifier.py
```

## Repository Structure

```text
Brake_Fade_GradientBoosting/
├── src/
│   └── train_classifier.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Engineering Interpretation

The model demonstrates how thermal and operating-state features can be organized into a multiclass early-warning pipeline. Real deployment would require measured brake-system data, sensor calibration, domain-shift testing, uncertainty analysis, and safety validation.
