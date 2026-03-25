# INSTRUCCIONES:
# 1. Sube este archivo a Google Colab como un notebook nuevo
# 2. Sube el conjunto de datos
# ============================================================

# ─────────────────────────────────────────────
# CELDA 1 — Instalación de dependencias
# ─────────────────────────────────────────────
# !pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn openpyxl joblib -q

# ─────────────────────────────────────────────
# CELDA 2 — Imports
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("Dependencias importadas correctamente")

# ─────────────────────────────────────────────
# CELDA 3 — Configuración global
# ─────────────────────────────────────────────
# Variables con data leakage (EXCLUIR del modelo)
LEAKAGE_VARS = [
    "clasfinal", "tiene_fec_hos", "pac_hos_", "fec_hos",
    "hipotensio", "choque", "hipotermia", "hem_mucosa",
    "hemorr_hem", "aum_hemato", "caida_plaq", "acum_liqui",
    "extravasac", "daño_organ"
]

# Variables permitidas para predicción (síntomas tempranos)
FEATURE_VARS = [
    "fiebre", "cefalea", "dolrretroo", "malgias",
    "artralgia", "erupcionr", "dolor_abdo", "vomito",
    "diarrea", "somnolenci", "hepatomeg"
]

# Variables solo para análisis/visualización
ANALYSIS_VARS = ["edad_", "sexo__M", "estrato_", "cod_mun_r", "fec_not"]

TARGET = "dengue_grave"
COD_VALLEDUPAR = 1  # Ajusta si el código real es diferente

THRESHOLDS = [0.3, 0.4, 0.5]
RANDOM_STATE = 42

print("Configuración global lista")

# ─────────────────────────────────────────────
# CELDA 4 — Carga y exploración inicial
# ─────────────────────────────────────────────
df_raw = pd.read_excel("DatasetParaModelar.xlsx")
print(f" Shape original: {df_raw.shape}")
print(f"\n Distribución de {TARGET}:")
print(df_raw[TARGET].value_counts())
print(f"\n   % dengue grave: {df_raw[TARGET].mean()*100:.2f}%")
print(f"\n Columnas disponibles:")
print(df_raw.columns.tolist())

# ─────────────────────────────────────────────
# CELDA 5 — Preprocesamiento y Feature Engineering
# ─────────────────────────────────────────────
df = df_raw.copy()

# Parsear fecha de notificación
df["fec_not"] = pd.to_datetime(df["fec_not"], errors="coerce")
df["anio"]    = df["fec_not"].dt.year
df["mes_not"] = df["fec_not"].dt.month

# Eliminar otras fechas para evitar leakage temporal
date_cols = [c for c in df.columns if c.startswith("fec_") and c != "fec_not"]
date_cols += ["fec_con_", "ini_sin_"]
df.drop(columns=[c for c in date_cols if c in df.columns], inplace=True, errors="ignore")

# Eliminar variables con data leakage
df.drop(columns=[c for c in LEAKAGE_VARS if c in df.columns], inplace=True, errors="ignore")

print(" Feature Engineering completado")
print(f"   Shape tras limpieza: {df.shape}")

# ─────────────────────────────────────────────
# CELDA 6 — EDA y Gráficos históricos
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(" Análisis Histórico — Dengue (Cesar & Valledupar)", fontsize=15, fontweight="bold")

# 1. Distribución del target
ax = axes[0, 0]
counts = df[TARGET].value_counts()
bars = ax.bar(["Dengue (0)", "Dengue Grave (1)"], counts.values,
              color=["#4C9BE8", "#E85C4C"], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,}", ha="center", fontweight="bold")
ax.set_title("Distribución de Dengue Grave")
ax.set_ylabel("Casos")

# 2. Casos por año
ax = axes[0, 1]
if "anio" in df.columns:
    anual = df.groupby("anio")[TARGET].agg(["sum", "count"]).reset_index()
    anual.columns = ["anio", "graves", "total"]
    ax.bar(anual["anio"], anual["total"], color="#4C9BE8", alpha=0.7, label="Total")
    ax.bar(anual["anio"], anual["graves"], color="#E85C4C", alpha=0.9, label="Graves")
    ax.set_title("Casos Totales vs Graves por Año")
    ax.set_xlabel("Año"); ax.set_ylabel("Casos")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# 3. Casos por edad
ax = axes[0, 2]
if "edad_" in df.columns:
    bins = [0, 5, 12, 18, 30, 45, 60, 100]
    labels = ["0-4", "5-12", "13-17", "18-29", "30-44", "45-59", "60+"]
    df["grupo_edad"] = pd.cut(df["edad_"], bins=bins, labels=labels, right=False)
    edad_data = df.groupby("grupo_edad")[TARGET].agg(["sum", "count"]).reset_index()
    edad_data.columns = ["grupo_edad", "graves", "total"]
    x = np.arange(len(edad_data))
    ax.bar(x - 0.2, edad_data["total"], 0.4, color="#4C9BE8", alpha=0.8, label="Total")
    ax.bar(x + 0.2, edad_data["graves"], 0.4, color="#E85C4C", alpha=0.9, label="Graves")
    ax.set_xticks(x); ax.set_xticklabels(edad_data["grupo_edad"], rotation=30)
    ax.set_title("Casos por Grupo de Edad")
    ax.set_ylabel("Casos"); ax.legend()

# 4. Casos por estrato
ax = axes[1, 0]
if "estrato_" in df.columns:
    estrato_data = df.groupby("estrato_")[TARGET].agg(["sum", "count"]).reset_index()
    estrato_data.columns = ["estrato", "graves", "total"]
    estrato_data = estrato_data[estrato_data["estrato"].between(1, 6)]
    x = np.arange(len(estrato_data))
    ax.bar(x - 0.2, estrato_data["total"], 0.4, color="#4C9BE8", alpha=0.8, label="Total")
    ax.bar(x + 0.2, estrato_data["graves"], 0.4, color="#E85C4C", alpha=0.9, label="Graves")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Estrato {int(e)}" for e in estrato_data["estrato"]])
    ax.set_title("Casos por Estrato Socioeconómico")
    ax.set_ylabel("Casos"); ax.legend()

# 5. Casos por sexo
ax = axes[1, 1]
if "sexo__M" in df.columns:
    sexo_data = df.groupby("sexo__M")[TARGET].agg(["sum", "count"]).reset_index()
    sexo_data.columns = ["sexo_M", "graves", "total"]
    labels_sexo = ["Femenino" if s == 0 else "Masculino" for s in sexo_data["sexo_M"]]
    x = np.arange(len(labels_sexo))
    ax.bar(x - 0.2, sexo_data["total"], 0.4, color="#4C9BE8", alpha=0.8, label="Total")
    ax.bar(x + 0.2, sexo_data["graves"], 0.4, color="#E85C4C", alpha=0.9, label="Graves")
    ax.set_xticks(x); ax.set_xticklabels(labels_sexo)
    ax.set_title("Casos por Sexo")
    ax.set_ylabel("Casos"); ax.legend()

# 6. Síntomas más frecuentes en dengue grave
ax = axes[1, 2]
symptom_cols = [c for c in FEATURE_VARS if c in df.columns]
symptom_prev = df[df[TARGET] == 1][symptom_cols].mean().sort_values(ascending=True)
symptom_labels = {
    "fiebre": "Fiebre", "cefalea": "Cefalea", "dolrretroo": "Dolor retro-ocular",
    "malgias": "Mialgias", "artralgia": "Artralgias", "erupcionr": "Erupción",
    "dolor_abdo": "Dolor abdominal", "vomito": "Vómito", "diarrea": "Diarrea",
    "somnolenci": "Somnolencia", "hepatomeg": "Hepatomegalia"
}
ax.barh(
    [symptom_labels.get(s, s) for s in symptom_prev.index],
    symptom_prev.values * 100,
    color="#E85C4C", alpha=0.85
)
ax.set_title("Prevalencia de Síntomas en Dengue Grave")
ax.set_xlabel("% de casos")

plt.tight_layout()
plt.savefig("historico_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráficos EDA guardados → historico_eda.png")

# ─────────────────────────────────────────────
# CELDA 7 — Exportar CSVs históricos para Streamlit
# ─────────────────────────────────────────────
# Histórico anual (Cesar completo)
if "anio" in df.columns:
    anual_cesar = df.groupby("anio").agg(
        total=(TARGET, "count"),
        graves=(TARGET, "sum")
    ).reset_index()
    anual_cesar["casos_anuales"] = anual_cesar["total"].diff().fillna(anual_cesar["total"])
    anual_cesar["graves_anuales"] = anual_cesar["graves"].diff().fillna(anual_cesar["graves"])
    anual_cesar.to_csv("historico_anual.csv", index=False)

    # Valledupar
    if "cod_mun_r" in df.columns:
        df_vdup = df[df["cod_mun_r"] == COD_VALLEDUPAR]
        anual_vdup = df_vdup.groupby("anio").agg(
            total=(TARGET, "count"),
            graves=(TARGET, "sum")
        ).reset_index()
        anual_vdup["casos_anuales"] = anual_vdup["total"].diff().fillna(anual_vdup["total"])
        anual_vdup["graves_anuales"] = anual_vdup["graves"].diff().fillna(anual_vdup["graves"])
        anual_vdup.to_csv("historico_anual_valledupar.csv", index=False)

# Histórico por edad
if "edad_" in df.columns:
    df.groupby("grupo_edad")[TARGET].agg(["sum", "count"]).reset_index().to_csv("historico_edad.csv", index=False)

# Histórico por estrato
if "estrato_" in df.columns:
    df[df["estrato_"].between(1, 6)].groupby("estrato_")[TARGET].agg(["sum", "count"]).reset_index().to_csv("historico_estrato.csv", index=False)

# Histórico por sexo
if "sexo__M" in df.columns:
    df.groupby("sexo__M")[TARGET].agg(["sum", "count"]).reset_index().to_csv("historico_sexo.csv", index=False)

print(" CSVs históricos exportados")

# ─────────────────────────────────────────────
# CELDA 8 — Preparación de datos para modelado
# ─────────────────────────────────────────────
available_features = [f for f in FEATURE_VARS if f in df.columns]
print(f" Features disponibles ({len(available_features)}): {available_features}")

X = df[available_features].copy()
y = df[TARGET].copy()

# Imputar nulos con la moda (variables binarias)
X = X.fillna(X.mode().iloc[0])

# Split estratificado 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f" Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Positivos en train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

# ─────────────────────────────────────────────
# CELDA 9 — SMOTE + Escalado
# ─────────────────────────────────────────────
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f" Tras SMOTE — Train: {X_train_res.shape} | Positivos: {y_train_res.sum()}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# Guardar features y scaler
features = X.columns.tolist()
joblib.dump(features, "features.pkl")
joblib.dump(scaler,   "scaler.pkl")
print(" scaler.pkl y features.pkl guardados")

# ─────────────────────────────────────────────
# CELDA 10 — Entrenamiento de modelos
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_STATE, n_iter_no_change=15
    )
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train_res)
    trained_models[name] = model
    print(f" {name} entrenado")

# ─────────────────────────────────────────────
# CELDA 11 — Evaluación con múltiples thresholds
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_test_sc, y_test, thresholds=THRESHOLDS):
    probs = model.predict_proba(X_test_sc)[:, 1]
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc  = average_precision_score(y_test, probs)
    results = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        results.append({
            "modelo": name, "threshold": thr,
            "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc":   round(roc_auc, 4),
            "pr_auc":    round(pr_auc, 4),
        })
    return results

all_results = []
for name, model in trained_models.items():
    all_results.extend(evaluate_model(name, model, X_test_sc, y_test))

df_metrics = pd.DataFrame(all_results)
print("\n Tabla de Métricas por Modelo y Threshold:")
print(df_metrics.to_string(index=False))
df_metrics.to_csv("metricas_modelos.csv", index=False)
print("\n metricas_modelos.csv guardado")

# ─────────────────────────────────────────────
# CELDA 12 — Selección del mejor modelo
# ─────────────────────────────────────────────
# Prioridad: recall alto + f1 razonable
best_row = df_metrics.sort_values(["recall", "f1"], ascending=False).iloc[0]
best_model_name = best_row["modelo"]
best_threshold  = best_row["threshold"]
best_model      = trained_models[best_model_name]

print(f"\n Mejor modelo: {best_model_name} | Threshold: {best_threshold}")
print(f"   Recall: {best_row['recall']} | Precision: {best_row['precision']} | F1: {best_row['f1']}")

joblib.dump(best_model, "modelo.pkl")
print(" modelo.pkl guardado")

# ─────────────────────────────────────────────
# CELDA 13 — Gráficos de evaluación del modelo
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f" Evaluación del Mejor Modelo — {best_model_name}", fontsize=13, fontweight="bold")

probs_best = best_model.predict_proba(X_test_sc)[:, 1]
preds_best = (probs_best >= best_threshold).astype(int)

# Matriz de confusión
cm = confusion_matrix(y_test, preds_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Dengue", "Grave"], yticklabels=["Dengue", "Grave"])
axes[0].set_title("Matriz de Confusión")
axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Real")

# Curva ROC — todos los modelos
for name, model in trained_models.items():
    pr = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pr)
    auc = roc_auc_score(y_test, pr)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("Curva ROC")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].legend(fontsize=8)

# Curva Precision-Recall — todos los modelos
for name, model in trained_models.items():
    pr = model.predict_proba(X_test_sc)[:, 1]
    precision_c, recall_c, _ = precision_recall_curve(y_test, pr)
    ap = average_precision_score(y_test, pr)
    axes[2].plot(recall_c, precision_c, label=f"{name} (AP={ap:.3f})")
axes[2].set_title("Curva Precision-Recall")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("evaluacion_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print(" evaluacion_modelos.png guardado")

# Comparación de métricas entre modelos (barras)
fig, ax = plt.subplots(figsize=(12, 5))
best_per_model = df_metrics.sort_values("recall", ascending=False).groupby("modelo").first().reset_index()
x_pos = np.arange(len(best_per_model))
width = 0.2
for i, metric in enumerate(["recall", "precision", "f1", "roc_auc"]):
    ax.bar(x_pos + i * width, best_per_model[metric], width, label=metric.upper())
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(best_per_model["modelo"], rotation=10)
ax.set_title("Comparación de Modelos — Mejor Threshold por Modelo")
ax.legend(); ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print(" comparacion_modelos.png guardado")

# ─────────────────────────────────────────────
# CELDA 14 — Predicciones futuras (proyección 5 años)
# ─────────────────────────────────────────────
def generar_proyeccion(df_historico, region_name, n_years=5):
    """Proyección lineal simple basada en tendencia histórica."""
    if df_historico.empty or "anio" not in df_historico.columns:
        return pd.DataFrame()
    df_h = df_historico.copy()
    # Usar casos anuales (no acumulados)
    df_h["casos"] = df_h["total"].diff().fillna(df_h["total"])
    ultimo_anio  = int(df_h["anio"].max())
    media_casos  = df_h["casos"].tail(3).mean()
    std_casos    = df_h["casos"].tail(3).std()
    media_graves = df_h["graves"].tail(3).mean() if "graves" in df_h.columns else 0
    rows = []
    for i in range(1, n_years + 1):
        rows.append({
            "anio": ultimo_anio + i,
            "casos_proyectados": max(0, round(media_casos + np.random.normal(0, std_casos * 0.3))),
            "graves_proyectados": max(0, round(media_graves)),
            "region": region_name,
            "tipo": "proyeccion"
        })
    return pd.DataFrame(rows)

# Cargar CSVs generados antes
try:
    hist_cesar  = pd.read_csv("historico_anual.csv")
    proy_cesar  = generar_proyeccion(hist_cesar,  "Cesar")

    hist_vdup   = pd.read_csv("historico_anual_valledupar.csv")
    proy_vdup   = generar_proyeccion(hist_vdup, "Valledupar")

    proy_cesar.to_csv("predicciones_cesar.csv", index=False)
    proy_vdup.to_csv("predicciones_valledupar.csv", index=False)
    print(" predicciones_cesar.csv y predicciones_valledupar.csv guardados")

    # Gráfico de proyecciones
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(" Proyección de Casos — Próximos 5 Años", fontsize=13, fontweight="bold")

    for ax, hist, proy, title in zip(
        axes,
        [hist_cesar, hist_vdup],
        [proy_cesar, proy_vdup],
        ["Cesar (Departamento)", "Valledupar (Ciudad)"]
    ):
        hist_plot = hist.copy()
        hist_plot["casos"] = hist_plot["total"].diff().fillna(hist_plot["total"])
        ax.plot(hist_plot["anio"], hist_plot["casos"],
                "o-", color="#4C9BE8", label="Histórico Total", linewidth=2)
        if "graves" in hist_plot.columns:
            ax.plot(hist_plot["anio"], hist_plot["graves"],
                    "s-", color="#E85C4C", label="Histórico Graves", linewidth=2)
        ax.plot(proy["anio"], proy["casos_proyectados"],
                "o--", color="#4C9BE8", alpha=0.6, label="Proyección Total")
        ax.plot(proy["anio"], proy["graves_proyectados"],
                "s--", color="#E85C4C", alpha=0.6, label="Proyección Graves")
        ax.axvline(x=hist_plot["anio"].max(), color="gray", linestyle=":", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Año"); ax.set_ylabel("Casos")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("proyecciones_futuras.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(" proyecciones_futuras.png guardado")
except Exception as e:
    print(f"  Error en proyecciones: {e}")

# ─────────────────────────────────────────────
# CELDA 15 — Verificación debug final
# ─────────────────────────────────────────────
print("\n DEBUG — Verificación final")
print("Features usadas en el modelo:")
print(joblib.load("features.pkl"))

print("\nPrueba de predicción con input ficticio:")
sample_input = {f: 1 for f in features}  # todos los síntomas presentes
input_df = pd.DataFrame([sample_input])
input_df = input_df[features]  # asegurar orden correcto
print(input_df)

input_sc  = scaler.transform(input_df)
prob_test = best_model.predict_proba(input_sc)[0][1]
print(f"\n Probabilidad dengue grave (input de prueba): {prob_test:.4f}")

print("\n🎉 PIPELINE COMPLETADO — Artefactos generados:")
print("  modelo.pkl, scaler.pkl, features.pkl")
print("  historico_anual.csv, historico_anual_valledupar.csv")
print("  historico_edad.csv, historico_estrato.csv, historico_sexo.csv")
print("  predicciones_cesar.csv, predicciones_valledupar.csv")
print("  metricas_modelos.csv")
print("  historico_eda.png, evaluacion_modelos.png")
print("  comparacion_modelos.png, proyecciones_futuras.png")
