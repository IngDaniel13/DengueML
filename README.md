# 🦟 Sistema de Predicción de Dengue Grave
## Departamento del Cesar — Colombia

---

## 📁 Archivos del Proyecto

```
dengue_project/
├── dengue_colab.py       ← Código para Google Colab (pipeline completo)
├── streamlit_app.py      ← App Streamlit (dashboard + predicción)
├── requirements.txt      ← Dependencias de Streamlit
└── README.md             ← Este archivo
```

---

## 🚀 PASO 1 — Ejecutar el Pipeline en Google Colab

### A. Abrir Colab
Ve a https://colab.research.google.com y crea un notebook nuevo.

### B. Subir el dataset
En el panel izquierdo (ícono de carpeta 📁) sube:
- `DatasetParaModelar.xlsx`

### C. Pegar el código
Abre `dengue_colab.py` y copia TODO el contenido.
En Colab, crea las celdas separando por los bloques marcados con:
```
# ─────────────────────────────────────────────
# CELDA N — Descripción
# ─────────────────────────────────────────────
```
Cada bloque es una celda independiente.

> 💡 Tip: Puedes pegar todo en una sola celda si prefieres ejecutar de una vez.

### D. Instalar dependencias (primera celda)
Descomenta la línea:
```python
# !pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn openpyxl joblib -q
```
Quítale el `#` y ejecútala.

### E. Ejecutar todas las celdas
Runtime > Run all (o Ctrl+F9)

### F. Descargar los artefactos generados
Al finalizar, descarga TODOS estos archivos desde el panel de archivos de Colab:

**Modelos:**
- `modelo.pkl`
- `scaler.pkl`
- `features.pkl`

**CSVs históricos:**
- `historico_anual.csv`
- `historico_anual_valledupar.csv`
- `historico_edad.csv`
- `historico_estrato.csv`
- `historico_sexo.csv`

**CSVs predicciones:**
- `predicciones_cesar.csv`
- `predicciones_valledupar.csv`
- `metricas_modelos.csv`

**Gráficos (opcionales, ya se generan en Streamlit):**
- `historico_eda.png`
- `evaluacion_modelos.png`
- `comparacion_modelos.png`
- `proyecciones_futuras.png`

---

## 🚀 PASO 2 — Ejecutar la App Streamlit

### A. Organizar carpeta local
Crea una carpeta y pon TODOS estos archivos juntos:
```
mi_carpeta/
├── streamlit_app.py
├── requirements.txt
├── modelo.pkl
├── scaler.pkl
├── features.pkl
├── historico_anual.csv
├── historico_anual_valledupar.csv
├── historico_edad.csv
├── historico_estrato.csv
├── historico_sexo.csv
├── predicciones_cesar.csv
├── predicciones_valledupar.csv
└── metricas_modelos.csv
```

### B. Instalar dependencias
```bash
pip install -r requirements.txt
```

### C. Ejecutar Streamlit
```bash
streamlit run streamlit_app.py
```
Se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📊 Secciones de la App

| Sección | Descripción |
|---------|-------------|
| 📊 Análisis Histórico | Gráficos de casos por año, edad, estrato y sexo. Filtra por Cesar o Valledupar. |
| 🔮 Predicción Individual | Formulario con síntomas. Muestra probabilidad de dengue grave con gauge visual. |
| 📈 Proyección Futura | Proyección a 5 años con línea histórica + línea proyectada separadas por región. |
| 📋 Métricas del Modelo | Tabla comparativa de Logistic Regression, Random Forest y MLP con heatmap de métricas. |

---

## ⚠️ Notas Importantes

- El modelo usa SOLO síntomas clínicos tempranos. No incluye variables de resultado (sin data leakage).
- El **Recall** es la métrica prioritaria: más importante detectar casos graves que evitar falsas alarmas.
- Las proyecciones son estimaciones basadas en tendencia histórica. No son pronóstico oficial.
- Ajusta `COD_VALLEDUPAR = 20001` en el notebook si el código real del municipio es diferente.

---

## 🛠️ Solución de Problemas

**Error: "No se pudieron cargar los artefactos"**
→ Asegúrate que `modelo.pkl`, `scaler.pkl` y `features.pkl` estén en la misma carpeta que `streamlit_app.py`.

**Gráficos vacíos en histórico**
→ Los CSVs históricos no están en la carpeta. Descárgalos de Colab.

**SMOTE no instala**
→ Ejecuta: `pip install imbalanced-learn --upgrade`

**Columnas no encontradas en el dataset**
→ Verifica que tu Excel tenga exactamente las columnas especificadas. El script es robusto a columnas faltantes pero necesita al menos las variables de síntomas.
