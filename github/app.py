"""
Sistema de Alerta Temprana de Readmisión Hospitalaria
======================================================
Fase 4 del portfolio: App interactiva con Streamlit

Deploy: streamlit run app.py
Deploy en la nube: https://streamlit.io/cloud (gratuito)

Dependencias:
    pip install streamlit pandas scikit-learn xgboost shap plotly joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import joblib
import os

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadmissionAlert · Hospital Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# ESTILOS CUSTOM
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tipografía y base */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label { 
        font-size: 12px; 
        text-transform: uppercase; 
        letter-spacing: .05em;
        color: #888 !important;
    }

    /* Tarjetas de riesgo */
    .risk-card {
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,.08);
    }
    .risk-high   { background: linear-gradient(135deg, #3d0f0f 0%, #1a0606 100%); border-color: #c0392b44; }
    .risk-medium { background: linear-gradient(135deg, #3d2f0f 0%, #1a1206 100%); border-color: #e67e2244; }
    .risk-low    { background: linear-gradient(135deg, #0f3d1a 0%, #061a0b 100%); border-color: #27ae6044; }

    .risk-label  { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; opacity: .6; margin-bottom: 4px; }
    .risk-value  { font-size: 42px; font-weight: 300; font-family: 'IBM Plex Mono', monospace; line-height: 1; }
    .risk-badge  { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; margin-top: 8px; }
    .badge-high   { background: #c0392b33; color: #e74c3c; border: 1px solid #c0392b55; }
    .badge-medium { background: #e67e2233; color: #f39c12; border: 1px solid #e67e2255; }
    .badge-low    { background: #27ae6033; color: #2ecc71; border: 1px solid #27ae6055; }

    /* Barra de probabilidad */
    .prob-bar-wrap { background: #1a1a2e; border-radius: 6px; height: 8px; margin: 12px 0 4px; overflow: hidden; }
    .prob-bar      { height: 100%; border-radius: 6px; transition: width .5s ease; }

    /* SHAP cards */
    .shap-card { 
        background: #141620; 
        border: 1px solid #2a2d3e; 
        border-radius: 8px; 
        padding: 12px 16px; 
        margin-bottom: 8px;
        display: flex; align-items: center; gap: 12px;
    }
    .shap-name  { font-size: 13px; color: #c0c0c0; flex: 1; }
    .shap-value { font-size: 12px; font-family: 'IBM Plex Mono', monospace; color: #888; width: 60px; text-align:right; }
    .shap-bar-pos { background: #c0392b; border-radius: 3px; height: 6px; }
    .shap-bar-neg { background: #2980b9; border-radius: 3px; height: 6px; }

    /* Títulos de sección */
    .section-header { 
        font-size: 11px; 
        text-transform: uppercase; 
        letter-spacing: .12em; 
        color: #888; 
        border-bottom: 1px solid #2a2d3e; 
        padding-bottom: 8px; 
        margin: 20px 0 14px; 
    }

    /* Ocultar elementos de Streamlit que no queremos */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CARGA DEL MODELO
# Asume que entrenaste y guardaste el modelo con:
#   joblib.dump(pipeline, "model/pipeline.joblib")
#   joblib.dump(feature_names, "model/feature_names.joblib")
# Ver fase3_baseline.py y el script de entrenamiento XGBoost
# ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo entrenado. 
    Si no existe, usa un modelo de demo para que la app funcione igual.
    """
    model_path = "model/pipeline.joblib"
    features_path = "model/feature_names.joblib"

    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return pipeline, feature_names
    else:
        # Modelo de demo (para testear la UI sin haber entrenado aún)
        return None, None


pipeline, feature_names = cargar_modelo()
DEMO_MODE = pipeline is None


def predecir(datos_paciente: dict) -> tuple[float, dict]:
    """
    Retorna (probabilidad_readmision, shap_values_top5).
    En modo demo, genera valores simulados coherentes.
    """
    if DEMO_MODE:
        # Simulación para desarrollo de la UI
        base = 0.15
        if datos_paciente.get("number_inpatient", 0) > 1:
            base += 0.25
        if datos_paciente.get("A1Cresult") in [">7", ">8"]:
            base += 0.15
        if datos_paciente.get("time_in_hospital", 3) > 7:
            base += 0.12
        if datos_paciente.get("discharge_disposition_id") == "Alta a domicilio":
            base -= 0.05
        prob = min(0.95, max(0.03, base + np.random.normal(0, 0.02)))

        shap_demo = {
            "Internaciones previas":     +0.18 if datos_paciente.get("number_inpatient", 0) > 1 else -0.04,
            "HbA1c (A1C result)":        +0.14 if datos_paciente.get("A1Cresult") in [">7",">8"] else -0.06,
            "Días de internación":       +0.11 if datos_paciente.get("time_in_hospital", 3) > 7 else -0.03,
            "Cambio en medicación":      +0.09 if datos_paciente.get("change") == "Sí" else -0.02,
            "Cantidad de diagnósticos":  +0.07 if datos_paciente.get("number_diagnoses", 5) > 7 else -0.01,
        }
        return round(prob, 4), shap_demo

    # Modelo real
    # ─────────────────────────────────────────────────────
    # El formulario manda strings categóricos crudos.
    # Hay que aplicar get_dummies igual que en el entrenamiento
    # y luego alinear las columnas con las que vio el modelo.
    # ─────────────────────────────────────────────────────

    # 1. Crear DataFrame con los datos del formulario
    df_raw = pd.DataFrame([datos_paciente])

    # 2. Separar columnas numéricas y categóricas
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
    cols_numericas   = [c for c in cols_numericas   if c in df_raw.columns]
    cols_categoricas = [c for c in cols_categoricas if c in df_raw.columns]

    # 3. Aplicar get_dummies igual que en el entrenamiento
    df_encoded = pd.get_dummies(df_raw[cols_categoricas + cols_numericas], drop_first=True)

    # 4. Alinear columnas — agregar las que faltan con 0, sacar las que sobran
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 5. Predecir
    prob = pipeline.predict_proba(df_encoded)[0][1]

    # 6. SHAP
    explainer = shap.TreeExplainer(pipeline.named_steps["modelo"])
    X_transformed = pipeline.named_steps["scaler"].transform(df_encoded)
    shap_vals = explainer.shap_values(X_transformed)[0]
    shap_dict = dict(zip(feature_names, shap_vals))
    top5 = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    return round(float(prob), 4), top5


def nivel_riesgo(prob: float) -> tuple[str, str, str]:
    """Retorna (nivel, clase_css, clase_badge)"""
    if prob >= 0.50:
        return "ALTO", "risk-high", "badge-high"
    elif prob >= 0.25:
        return "MEDIO", "risk-medium", "badge-medium"
    else:
        return "BAJO", "risk-low", "badge-low"


# ─────────────────────────────────────────────────────────
# SIDEBAR — FORMULARIO DEL PACIENTE
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 ReadmissionAlert")
    st.markdown("<div style='font-size:12px;color:#666;margin-bottom:20px;'>Sistema de alerta temprana · Diabetes 130</div>", unsafe_allow_html=True)

    if DEMO_MODE:
        st.warning("Modo demo activo. Entrená el modelo para usar predicciones reales.", icon="⚠️")

    st.markdown("---")
    st.markdown("**Datos del paciente**")

    # Demografía
    edad = st.selectbox("Rango de edad", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
    genero = st.selectbox("Género", ["Masculino", "Femenino"])

    st.markdown("---")
    st.markdown("**Internación actual**")

    dias_internacion = st.slider("Días de internación", 1, 14, 4)
    num_lab = st.slider("Análisis realizados", 1, 100, 45)
    num_medicamentos = st.slider("Medicamentos administrados", 1, 81, 16)
    num_diagnosticos = st.slider("Cantidad de diagnósticos", 1, 16, 7)

    tipo_alta = st.selectbox("Tipo de alta", [
        "Alta a domicilio",
        "Alta a centro de rehabilitación",
        "Alta a otro hospital",
        "Alta con cuidados paliativos",
    ])

    st.markdown("---")
    st.markdown("**Historia clínica**")

    internaciones_prev = st.number_input("Internaciones previas (1 año)", 0, 10, 0)
    visitas_emergencia = st.number_input("Visitas a emergencias (1 año)", 0, 10, 0)
    visitas_ambulatorio = st.number_input("Consultas ambulatorias (1 año)", 0, 10, 0)

    st.markdown("---")
    st.markdown("**Laboratorio y medicación**")

    a1c = st.selectbox("Resultado HbA1c", ["None", "Norm", ">7", ">8"])
    glucosa_suero = st.selectbox("Glucosa en suero", ["None", "Norm", ">200", ">300"])
    insulina = st.selectbox("Insulina", ["No", "Down", "Steady", "Up"])
    cambio_med = st.selectbox("Cambio en medicación", ["No", "Sí"])
    med_diabetes = st.selectbox("Medicación para diabetes", ["Sí", "No"])

    st.markdown("---")
    analizar = st.button("🔍 Analizar riesgo", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────
col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown("## Sistema de alerta temprana de readmisión")
    st.markdown("<div style='color:#888;font-size:14px;margin-top:-8px;'>Predicción de readmisión hospitalaria en &lt;30 días · Pacientes diabéticos</div>", unsafe_allow_html=True)
with col_info:
    st.markdown("<div style='text-align:right;font-size:12px;color:#555;margin-top:12px;'>Dataset: UCI Diabetes 130<br>Modelo: XGBoost + SHAP</div>", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────
# ESTADO INICIAL (antes de analizar)
# ─────────────────────────────────────────────────────────
if not analizar:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#555;'>
        <div style='font-size:48px;margin-bottom:16px;'>←</div>
        <div style='font-size:16px;'>Completá los datos del paciente en el panel lateral<br>y presioná <strong style="color:#e0e0e0">Analizar riesgo</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────
datos = {
    "age":                     edad,
    "gender":                  genero,
    "time_in_hospital":        dias_internacion,
    "num_lab_procedures":      num_lab,
    "num_medications":         num_medicamentos,
    "number_diagnoses":        num_diagnosticos,
    "discharge_disposition_id": tipo_alta,
    "number_inpatient":        internaciones_prev,
    "number_emergency":        visitas_emergencia,
    "number_outpatient":       visitas_ambulatorio,
    "A1Cresult":               a1c,
    "max_glu_serum":           glucosa_suero,
    "insulin":                 insulina,
    "change":                  cambio_med,
    "diabetesMed":             med_diabetes,
}

with st.spinner("Calculando riesgo..."):
    prob, shap_top5 = predecir(datos)

nivel, clase_card, clase_badge = nivel_riesgo(prob)
color_barra = {"ALTO": "#e74c3c", "MEDIO": "#f39c12", "BAJO": "#2ecc71"}[nivel]


# ─────────────────────────────────────────────────────────
# RESULTADO PRINCIPAL
# ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="risk-card {clase_card}">
        <div class="risk-label">Probabilidad de readmisión</div>
        <div class="risk-value">{prob*100:.1f}%</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{prob*100:.1f}%;background:{color_barra};"></div>
        </div>
        <span class="risk-badge {clase_badge}">Riesgo {nivel}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Comparación con umbral
    umbral = 0.35  # Obtenido de la optimización por F2-score en fase 3
    diferencia = prob - umbral
    st.markdown(f"""
    <div class="risk-card" style="background:#141620;border-color:#2a2d3e;">
        <div class="risk-label">Umbral clínico (F2-optimizado)</div>
        <div class="risk-value" style="color:#888;">{umbral*100:.0f}%</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{umbral*100:.0f}%;background:#444;"></div>
        </div>
        <span class="risk-badge" style="background:#1a1a2e;color:#888;border-color:#333;">
            {'Por encima' if diferencia > 0 else 'Por debajo'} del umbral ({abs(diferencia)*100:.1f}pp)
        </span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Decisión recomendada
    recomendaciones = {
        "ALTO": ("🔴 Intervención activa", "Contactar al equipo de gestión de casos. Programar seguimiento en 48-72hs post-alta. Revisar plan de medicación."),
        "MEDIO": ("🟡 Seguimiento reforzado", "Programar llamada de seguimiento a los 7 días. Verificar adherencia al tratamiento. Evaluar apoyo social."),
        "BAJO": ("🟢 Alta estándar", "Seguimiento ambulatorio de rutina. Entregar plan de autocuidado. Control en 30 días."),
    }
    titulo_rec, texto_rec = recomendaciones[nivel]
    st.markdown(f"""
    <div class="risk-card" style="background:#141620;border-color:#2a2d3e;">
        <div class="risk-label">Acción recomendada</div>
        <div style="font-size:15px;font-weight:500;margin:8px 0 10px;color:#e0e0e0;">{titulo_rec}</div>
        <div style="font-size:13px;color:#888;line-height:1.5;">{texto_rec}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# EXPLICABILIDAD — SHAP
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>¿Por qué este resultado? · Explicación SHAP</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:13px;color:#666;margin-bottom:16px;'>Los factores que más influyeron en la predicción para este paciente. Rojo = aumenta el riesgo · Azul = lo reduce.</div>", unsafe_allow_html=True)

col_shap, col_gauge = st.columns([3, 2])

with col_shap:
    max_abs = max(abs(v) for v in shap_top5.values()) if shap_top5 else 1

    for nombre, valor in sorted(shap_top5.items(), key=lambda x: abs(x[1]), reverse=True):
        ancho_pct = min(100, int(abs(valor) / max_abs * 100))
        color = "#c0392b" if valor > 0 else "#2980b9"
        signo = "▲" if valor > 0 else "▼"
        label_dir = "Aumenta riesgo" if valor > 0 else "Reduce riesgo"

        st.markdown(f"""
        <div class="shap-card">
            <div>
                <div style="font-size:10px;color:{color};text-transform:uppercase;letter-spacing:.05em;">{signo} {label_dir}</div>
                <div class="shap-name">{nombre}</div>
                <div style="margin-top:6px;background:#0d0f1a;border-radius:3px;height:6px;width:160px;overflow:hidden;">
                    <div style="width:{ancho_pct}%;background:{color};height:100%;border-radius:3px;"></div>
                </div>
            </div>
            <div class="shap-value">{valor:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)

with col_gauge:
    # Gráfico de gauge con Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e0e0e0", "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444", "tickfont": {"color": "#666", "size": 11}},
            "bar": {"color": color_barra, "thickness": 0.25},
            "bgcolor": "#0d0f1a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],   "color": "#0f3d1a"},
                {"range": [25, 50],  "color": "#3d2f0f"},
                {"range": [50, 100], "color": "#3d0f0f"},
            ],
            "threshold": {
                "line": {"color": "#888", "width": 2},
                "thickness": 0.75,
                "value": umbral * 100,
            },
        },
        title={"text": "Riesgo de readmisión", "font": {"color": "#888", "size": 12}},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        height=250,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# CONTEXTO DEL MODELO
# ─────────────────────────────────────────────────────────
with st.expander("ℹ️ Información del modelo y limitaciones"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Sobre el modelo**
        - Algoritmo: XGBoost con class_weight balanceado
        - Dataset: UCI Diabetes 130-US Hospitals (1999–2008)
        - Umbral: Optimizado por F2-score (penaliza falsos negativos x2)
        - Explicabilidad: SHAP TreeExplainer
        """)
    with col_b:
        st.markdown("""
        **Limitaciones importantes**
        - Entrenado en datos de hospitales de EE.UU. de hace >15 años
        - No reemplaza el juicio clínico del médico tratante
        - Puede presentar sesgos relacionados con raza/etnia en el dataset original
        - El umbral debe recalibrarse para cada institución
        """)
    st.markdown("""
    > Este sistema es una **herramienta de apoyo a la decisión clínica**, no un diagnóstico. 
    > La decisión final siempre recae en el equipo médico.
    """)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#444;'>Portfolio project · Data Science en Salud · "
    "Dataset: <a href='https://archive.ics.uci.edu/dataset/296' style='color:#555;'>UCI ML Repository</a></div>",
    unsafe_allow_html=True
)
