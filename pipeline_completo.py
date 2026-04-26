"""
pipeline_completo.py
=====================
Corre todo el proyecto en orden, en un solo archivo.
No hay dependencias entre scripts separados.

Ejecutar:
    python pipeline_completo.py

Requiere:
    pip install pandas numpy scikit-learn xgboost shap matplotlib joblib
"""

import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, fbeta_score,
    roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier


# ═════════════════════════════════════════════════════════
# PASO 1 — CARGAR Y PREPARAR DATOS
# ═════════════════════════════════════════════════════════

print("=" * 55)
print("PASO 1 — Cargando datos")
print("=" * 55)

df = pd.read_csv("data/diabetic_data.csv")
df = df.replace("?", np.nan)
df["readmitido"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["encounter_id", "patient_nbr", "readmitted"], errors="ignore")

cols_numericas = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
cols_categoricas = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed",
]
cols_numericas   = [c for c in cols_numericas   if c in df.columns]
cols_categoricas = [c for c in cols_categoricas if c in df.columns]

for col in cols_numericas:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

X = pd.get_dummies(df[cols_categoricas + cols_numericas], drop_first=True)
y = df["readmitido"]

print(f"Dataset: {X.shape[0]} pacientes, {X.shape[1]} features")
print(f"Positivos: {y.sum()} ({y.mean():.1%})")

# Train/test split — mismo random_state para todos los modelos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaler compartido entre baseline y XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# ═════════════════════════════════════════════════════════
# PASO 2 — BASELINE: REGRESIÓN LOGÍSTICA
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 2 — Baseline: Regresión Logística")
print("=" * 55)

baseline = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=42
)
baseline.fit(X_train_scaled, y_train)

y_prob_lr = baseline.predict_proba(X_test_scaled)[:, 1]

roc_lr = roc_auc_score(y_test, y_prob_lr)
pr_lr  = average_precision_score(y_test, y_prob_lr)
rec_lr = recall_score(y_test, (y_prob_lr >= 0.5).astype(int))

print(f"ROC-AUC : {roc_lr:.4f}")
print(f"PR-AUC  : {pr_lr:.4f}")
print(f"Recall  : {rec_lr:.4f}")


# ═════════════════════════════════════════════════════════
# PASO 3 — MODELO PRINCIPAL: XGBOOST
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 3 — Modelo principal: XGBoost")
print("=" * 55)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Partición de validación para early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.1, random_state=42, stratify=y_train
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Árboles entrenados: {xgb.best_iteration + 1} (early stopping)")

y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

roc_xgb = roc_auc_score(y_test, y_prob_xgb)
pr_xgb  = average_precision_score(y_test, y_prob_xgb)
rec_xgb = recall_score(y_test, (y_prob_xgb >= 0.5).astype(int))

print(f"ROC-AUC : {roc_xgb:.4f}  (baseline: {roc_lr:.4f})")
print(f"PR-AUC  : {pr_xgb:.4f}  (baseline: {pr_lr:.4f})")
print(f"Recall  : {rec_xgb:.4f}  (baseline: {rec_lr:.4f})")


# ═════════════════════════════════════════════════════════
# PASO 4 — OPTIMIZACIÓN DEL UMBRAL
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 4 — Optimización del umbral (F2-score)")
print("=" * 55)

best_f2, umbral_opt = 0, 0.5
for u in np.arange(0.10, 0.90, 0.01):
    y_pred_u = (y_prob_xgb >= u).astype(int)
    if y_pred_u.sum() == 0:
        continue
    f2 = fbeta_score(y_test, y_pred_u, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2, umbral_opt = f2, u

y_pred_opt = (y_prob_xgb >= umbral_opt).astype(int)
print(f"Umbral óptimo : {umbral_opt:.2f}")
print(f"F2-score      : {best_f2:.4f}")
print(f"Recall        : {recall_score(y_test, y_pred_opt):.4f}")
print(f"Precision     : {precision_score(y_test, y_pred_opt, zero_division=0):.4f}")


# ═════════════════════════════════════════════════════════
# PASO 5 — VALIDACIÓN CRUZADA
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 5 — Validación cruzada (5-fold, Stratified)")
print("=" * 55)

pipeline_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", XGBClassifier(
        n_estimators=xgb.best_iteration + 1,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_cv = cross_val_score(pipeline_cv, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
pr_cv  = cross_val_score(pipeline_cv, X, y, cv=skf, scoring="average_precision", n_jobs=-1)

print(f"ROC-AUC: {roc_cv.mean():.4f} ± {roc_cv.std():.4f}")
print(f"PR-AUC:  {pr_cv.mean():.4f} ± {pr_cv.std():.4f}")
print(f"Estabilidad: {'✓ estable' if roc_cv.std() < 0.02 else '⚠ revisar sobreajuste'}")


# ═════════════════════════════════════════════════════════
# PASO 6 — SHAP
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 6 — SHAP: Explicabilidad del modelo")
print("=" * 55)

feature_names = list(X.columns)
explainer     = shap.TreeExplainer(xgb)
shap_values   = explainer.shap_values(X_test_scaled)

# Importancia global
importancia = pd.DataFrame({
    "feature":       feature_names,
    "shap_mean_abs": np.abs(shap_values).mean(axis=0),
}).sort_values("shap_mean_abs", ascending=False)

print("\nTop 10 variables más importantes (SHAP global):")
print(importancia.head(10).to_string(index=False))

# Gráfico global
plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=feature_names,
    plot_type="bar", max_display=10, show=False,
)
plt.title("Top 10 variables — SHAP global", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150, bbox_inches="tight")
print("Guardado: shap_global.png")
plt.close()

# Gráfico beeswarm
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=feature_names,
    plot_type="dot", max_display=10, show=False,
)
plt.title("Impacto por variable — SHAP beeswarm", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
print("Guardado: shap_beeswarm.png")
plt.close()

# Waterfall de un paciente de alto riesgo
probs_test = xgb.predict_proba(X_test_scaled)[:, 1]
idx_alto   = np.where(probs_test >= 0.70)[0][0]

explanation = shap.Explanation(
    values=shap_values[idx_alto],
    base_values=explainer.expected_value,
    data=X_test_scaled[idx_alto],
    feature_names=feature_names,
)
plt.figure()
shap.waterfall_plot(explanation, max_display=8, show=False)
plt.title(f"SHAP — Paciente de alto riesgo (prob={probs_test[idx_alto]:.1%})", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_paciente_alto.png", dpi=150, bbox_inches="tight")
print("Guardado: shap_paciente_alto.png")
plt.close()


# ═════════════════════════════════════════════════════════
# PASO 7 — GUARDAR MODELO
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PASO 7 — Guardando modelo")
print("=" * 55)

os.makedirs("model", exist_ok=True)

# Pipeline final entrenado sobre TODO el train (sin partición de validación)
pipeline_final = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", XGBClassifier(
        n_estimators=xgb.best_iteration + 1,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )),
])
pipeline_final.fit(X_train, y_train)

joblib.dump(pipeline_final,        "model/pipeline.joblib")
joblib.dump(feature_names,         "model/feature_names.joblib")
joblib.dump(float(umbral_opt),     "model/umbral_optimo.joblib")

print("model/pipeline.joblib      ✓")
print("model/feature_names.joblib ✓")
print("model/umbral_optimo.joblib ✓")


# ═════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("RESUMEN — para el README")
print("=" * 55)
print(f"""
| Métrica   | Baseline (LR) | XGBoost |
|-----------|---------------|---------|
| ROC-AUC   | {roc_lr:.3f}         | {roc_xgb:.3f}   |
| PR-AUC    | {pr_lr:.3f}         | {pr_xgb:.3f}   |
| Recall    | {rec_lr:.3f}         | {rec_xgb:.3f}   |

XGBoost — umbral optimizado : {umbral_opt:.2f}
XGBoost — validación cruzada: ROC-AUC {roc_cv.mean():.3f} ± {roc_cv.std():.3f}
""")
