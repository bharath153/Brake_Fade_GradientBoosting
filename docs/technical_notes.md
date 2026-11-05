# Technical Notes — Brake Fade Gradient-Boosting Classifier

## Purpose

This repository demonstrates multiclass brake-fade risk classification from synthetic thermal and operating-state variables.

## Synthetic Inputs

The generated feature set includes:
- front and rear rotor temperature;
- caliper temperature;
- pad wear;
- pedal force;
- deceleration;
- brake-application duration;
- vehicle speed;
- consecutive stops.

The class target is generated from a rule-based fade score with added random variation.

## Model

The implementation uses scikit-learn's `GradientBoostingClassifier`.

It does **not** use the XGBoost library and does not claim SHAP-based interpretation.

## Validation

The repository uses:
- stratified train/test splitting;
- stratified 5-fold cross-validation;
- accuracy and macro-F1;
- confusion matrix;
- one-vs-rest ROC curves.

## Limitations

This is not a safety-validated brake model. Real deployment would require measured brake temperatures, fade ground truth, vehicle-specific calibration, uncertainty analysis, fault-tolerance assessment and validation under representative duty cycles.
