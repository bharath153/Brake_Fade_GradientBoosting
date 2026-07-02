"""
Brake Fade Early Warning System — XGBoost + SHAP
=================================================
Classifies brake fade risk (None / Mild / Severe) from
thermal sensor & brake pedal signals on a high-performance vehicle.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import os, json

np.random.seed(2024)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Synthetic brake sensor dataset ──
N = 3000
rng = np.random.default_rng(2024)

rotor_temp_front   = rng.uniform(80,  700, N)   # °C
rotor_temp_rear    = rng.uniform(60,  550, N)   # °C
caliper_temp       = rng.uniform(70,  500, N)   # °C
pad_wear_pct       = rng.uniform(5,   100, N)   # %
pedal_force        = rng.uniform(50,  500, N)   # N
deceleration       = rng.uniform(0.1, 1.2, N)   # g
brake_app_duration = rng.uniform(0.5, 15,  N)   # s
speed_at_apply     = rng.uniform(30,  250, N)   # km/h
consecutive_stops  = rng.integers(1, 15, N)     # count

# Target: 0=No Fade, 1=Mild Fade, 2=Severe Fade
fade_score = (
    0.004 * rotor_temp_front
    + 0.003 * caliper_temp
    + 0.008 * pad_wear_pct
    - 0.05  * deceleration
    + 0.01  * consecutive_stops
    + 0.0015 * speed_at_apply
    + rng.normal(0, 0.15, N)
)
labels = np.where(fade_score < 2.5, 0, np.where(fade_score < 4.0, 1, 2))
label_names = ['No Fade', 'Mild Fade', 'Severe Fade']
print(f"Class distribution: {np.bincount(labels)}")

X = np.column_stack([rotor_temp_front, rotor_temp_rear, caliper_temp, pad_wear_pct,
                     pedal_force, deceleration, brake_app_duration, speed_at_apply, consecutive_stops])
feature_names = ['Rotor Temp Front (°C)', 'Rotor Temp Rear (°C)', 'Caliper Temp (°C)',
                 'Pad Wear (%)', 'Pedal Force (N)', 'Deceleration (g)',
                 'Brake Duration (s)', 'Speed at Apply (km/h)', 'Consecutive Stops']

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2,
                                                     stratify=labels, random_state=42)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                  max_depth=4, subsample=0.85, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
f1  = cross_val_score(clf, X, labels, cv=cv, scoring='f1_macro')
print(f"CV Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
print(f"CV F1-macro: {f1.mean():.4f} ± {f1.std():.4f}")
print(classification_report(y_test, y_pred, target_names=label_names))
json.dump({"CV_Accuracy": round(acc.mean(),4), "CV_F1_macro": round(f1.mean(),4)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Brake Fade Classifier — XGBoost Performance', fontsize=14, fontweight='bold')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix  |  Acc = {(cm.diagonal().sum()/cm.sum()):.3f}', fontsize=11)

# ROC curves
y_bin = label_binarize(y_test, classes=[0,1,2])
colors = ['#1976D2','#FF5722','#388E3C']
for i, (name, col) in enumerate(zip(label_names, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set_xlabel('False Positive Rate', fontsize=11); axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title('One-vs-Rest ROC Curves', fontsize=11); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_performance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Feature Importance
fi = clf.feature_importances_
si = fi.argsort()[::-1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([feature_names[i] for i in si[::-1]], fi[si[::-1]],
        color=['#D32F2F' if fi[i]==fi.max() else '#EF9A9A' for i in si[::-1]],
        edgecolor='white')
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Feature Importance — Brake Fade Early Warning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Rotor temp vs caliper temp scatter (colored by class)
colors_s = ['#2196F3','#FF9800','#F44336']
fig, ax = plt.subplots(figsize=(9, 6))
for c, col, name in zip([0,1,2], colors_s, label_names):
    mask = labels == c
    ax.scatter(rotor_temp_front[mask], caliper_temp[mask], s=12, alpha=0.4,
               color=col, label=name, edgecolors='none')
ax.set_xlabel('Front Rotor Temperature (°C)', fontsize=11)
ax.set_ylabel('Caliper Temperature (°C)', fontsize=11)
ax.set_title('Thermal Feature Space — Brake Fade Classes', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_feature_space.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Probability over a simulated track run
n_sim = 300
sim_temp = np.linspace(80, 650, n_sim)
sim_wear = np.linspace(15, 85, n_sim)
sim_X = np.column_stack([sim_temp, sim_temp*0.7, sim_temp*0.65, sim_wear,
                          np.ones(n_sim)*200, np.ones(n_sim)*0.8,
                          np.ones(n_sim)*5, np.ones(n_sim)*180,
                          np.clip(np.arange(n_sim)//30, 1, 10)])
sim_prob = clf.predict_proba(sim_X)
fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, col) in enumerate(zip(label_names, colors)):
    ax.plot(sim_temp, sim_prob[:,i], lw=2, color=col, label=name)
ax.set_xlabel('Front Rotor Temperature (°C)', fontsize=11)
ax.set_ylabel('Class Probability', fontsize=11)
ax.set_title('Brake Fade Risk Probability — Simulated Track Run', fontsize=13, fontweight='bold')
ax.axvline(500, color='red', ls=':', lw=1.5, label='Critical threshold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_risk_probability.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")
