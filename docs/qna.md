# Interview Q&A — Brake Fade Early Warning System — XGBoost
## Gradient Boosting Classifier (XGBoost-style) | Active Safety & Brake Systems

### Q1: What is brake fade exactly?

Brake fade is a reduction in braking effectiveness caused by overheating. There are two types: (1) Pad fade — the friction material degrades and outgasses at high temperatures, creating a vapour layer that reduces friction; (2) Fluid fade — the brake fluid boils and forms compressible vapour pockets in the hydraulic circuit, causing a 'spongy' pedal. Modern high-performance vehicles use vented/drilled rotors, high-temperature fluids (DOT 5.1), and ceramics/carbon-ceramic pads to delay fade onset.

---

### Q2: Why 3-class classification rather than regression?

The ultimate decision (warn driver, limit torque, or emergency intervention) is discrete. A 3-class design matches the actionable states: No action, issue a warning, trigger AEB intervention. Regression of a continuous fade score is possible but requires a downstream thresholding step that the classifier combines. The classifier also allows direct probability calibration (e.g., P(Severe) > 0.7 → trigger AEB).

---

### Q3: How do you handle class imbalance?

In real thermal event data, severe fade events are rare (~10-15% of events). Techniques include: class_weight='balanced' in scikit-learn, SMOTE oversampling (imbalanced-learn), cost-sensitive loss functions (higher penalty for missing severe fade), and ensemble methods that stratify sampling. In this project, stratified train/test split ensures consistent class ratios.

---

### Q4: What is the AUC score and what does 0.97 mean?

AUC (Area Under the ROC Curve) measures the probability that the classifier ranks a random positive example higher than a random negative example. 0.5 = random, 1.0 = perfect. An AUC of 0.97 for the severe fade class means the model nearly perfectly discriminates severe fade events from non-severe events — a clinically/safety-significant threshold.

---

### Q5: How would this integrate with a real vehicle's ECU?

The model would be compiled to C code (via sklearn-porter or ONNX runtime for microcontrollers), then integrated into the Electronic Brake Control Module (EBCM). Inputs are read from the CAN bus at 100 Hz. The classifier runs every 50 ms. Output classes trigger CANopen signals to the instrument cluster (warning lamp), powertrain ECU (torque reduction), and ESC module (intervention preparation).

---

