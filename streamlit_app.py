"""
DENGUE GRAVE - APP STREAMLIT CON GRÁFICOS INTERACTIVOS
========================================================
Requiere en la misma carpeta:
    modelo.pkl, scaler.pkl, features.pkl
    historico_anual.csv, historico_anual_valledupar.csv
    historico_edad.csv, historico_estrato.csv, historico_sexo.csv
    predicciones_cesar.csv, predicciones_valledupar.csv
    metricas_modelos.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Dengue - Sistema de Alerta Temprana",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONFIGURACIÓN DE TEMA (CLARO/OSCURO)
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def get_theme_colors():
    if st.session_state.theme == "dark":
        return {
            "bg_color": "#0f172a",
            "card_bg": "#1e293b",
            "text_color": "#f1f5f9",
            "text_secondary": "#94a3b8",
            "border_color": "#334155",
            "header_gradient": "linear-gradient(135deg, #0f172a, #1e1b4b)",
            "grid_color": "rgba(128,128,128,0.2)",
            "plot_bg": "rgba(0,0,0,0)",
            "paper_bg": "rgba(0,0,0,0)",
            "primary": "#3b82f6",
            "danger": "#ef4444",
            "warning": "#f59e0b",
            "success": "#22c55e"
        }
    else:
        return {
            "bg_color": "#f8fafc",
            "card_bg": "#ffffff",
            "text_color": "#1e293b",
            "text_secondary": "#475569",
            "border_color": "#e2e8f0",
            "header_gradient": "linear-gradient(135deg, #1e3a5f, #2563eb)",
            "grid_color": "#e2e8f0",
            "plot_bg": "rgba(0,0,0,0)",
            "paper_bg": "rgba(0,0,0,0)",
            "primary": "#2563eb",
            "danger": "#dc2626",
            "warning": "#d97706",
            "success": "#16a34a"
        }

def configurar_grafica_tema(fig, height=400):
    """Configura gráficas para que se adapten al tema actual"""
    colors = get_theme_colors()
    fig.update_layout(
        height=height,
        dragmode=False,
        hovermode='closest',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=12, color=colors["text_color"]),
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors["text_color"])
        ),
        title_font=dict(color=colors["text_color"]),
        xaxis=dict(
            title_font=dict(color=colors["text_secondary"]),
            tickfont=dict(color=colors["text_secondary"]),
            gridcolor=colors["grid_color"]
        ),
        yaxis=dict(
            title_font=dict(color=colors["text_secondary"]),
            tickfont=dict(color=colors["text_secondary"]),
            gridcolor=colors["grid_color"]
        ),
        modebar=dict(
            orientation='v',
            activecolor='#2563eb',
            add=['zoomIn2d', 'zoomOut2d', 'resetScale2d'],
            remove=['zoom', 'pan', 'select', 'lasso', 'orbitRotation', 'tableRotation']
        )
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


# CSS ADAPTADO AL TEMA
def apply_theme_css():
    colors = get_theme_colors()
    st.markdown(f"""
    <style>
    :root {{
        --bg-color: {colors["bg_color"]};
        --card-bg: {colors["card_bg"]};
        --text-color: {colors["text_color"]};
        --text-secondary: {colors["text_secondary"]};
        --border-color: {colors["border_color"]};
        --header-gradient: {colors["header_gradient"]};
        --primary: {colors["primary"]};
        --danger: {colors["danger"]};
        --warning: {colors["warning"]};
        --success: {colors["success"]};
    }}
    
    .stApp {{
        background-color: var(--bg-color);
    }}
    
    .stApp, .stApp * {{
        color: var(--text-color) !important;
    }}
    
    .stButton button, .st-emotion-cache-1v0mbdj button {{
        color: white !important;
    }}
    
    .theme-toggle-btn {{
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 999;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 50px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .theme-toggle-btn:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .section-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color) !important;
        border-left: 4px solid var(--primary);
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }}
    
    .metric-card {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .metric-card div:first-child {{
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-color) !important;
    }}
    .metric-card div:last-child {{
        color: var(--text-secondary) !important;
        font-size: 0.85rem;
        margin-top: 4px;
    }}
    
    .risk-card-low {{
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }}
    .risk-card-moderate {{
        background: linear-gradient(135deg, #f59e0b, #d97706);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }}
    .risk-card-high {{
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }}
    
    .info-box {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    .warning-box {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--warning);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    .danger-box {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--danger);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    .alert-red {{
        background: #ffeaea;
        border-left: 4px solid #e85c4c;
        padding: 12px 16px;
        border-radius: 6px;
    }}
    .alert-green {{
        background: #eafff2;
        border-left: 4px solid #28a745;
        padding: 12px 16px;
        border-radius: 6px;
    }}
    .alert-yellow {{
        background: #fffbea;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 6px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 4px;
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1rem;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 6px;
        color: var(--text-secondary) !important;
        background-color: var(--bg-color);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary) !important;
        color: white !important;
    }}
    
    .custom-header {{
        background: var(--header-gradient);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }}
    .custom-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }}
    .custom-header p {{
        color: #bdd7f7 !important;
        margin: 8px 0 0 0;
        font-size: 1rem;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }}
    
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg-color);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--primary);
        border-radius: 4px;
    }}
    
    @media (max-width: 768px) {{
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.75rem;
            padding: 6px 10px;
        }}
        .section-title {{
            font-size: 1.1rem;
        }}
        .custom-header {{
            padding: 20px;
        }}
        .custom-header h1 {{
            font-size: 1.5rem;
        }}
        .metric-card {{
            padding: 12px;
        }}
        .metric-card div:first-child {{
            font-size: 1.3rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


# CARGA DE ARCHIVOS
@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

COLOR_TOTAL = "#3b82f6"
COLOR_GRAVE = "#ef4444"
COLOR_PROY = "#60a5fa"
COLOR_PROY_G = "#f87171"
COLOR_WARNING = "#f59e0b" 


apply_theme_css()

# SIDEBAR CON BOTÓN DE CAMBIO DE TEMA
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Aedes_aegypti_CDC-Gathany.jpg/320px-Aedes_aegypti_CDC-Gathany.jpg",
             use_container_width=True)
    st.markdown("## Apariencia")
    
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    theme_text = "Modo Oscuro" if st.session_state.theme == "light" else "Modo Claro"
    if st.button(f"{theme_icon} {theme_text}", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 Modelo Activo")
    try:
        model, scaler, features = load_artifacts()
        st.info(f"**Modelo:** Logistic Regression\n**Features:** {len(features)}")
    except:
        st.warning("No hay modelo cargado")
    
    st.markdown("---")
    st.markdown("### 📊 Datos")
    st.markdown("Fuente: SIVIGILA (Datos históricos dengue - Cesar)")
    st.markdown("Última actualización: 2025")
    st.markdown("---")
    st.caption("🏥 Uso exclusivo para apoyo diagnóstico.\nNo reemplaza criterio médico.")

# ──────────────────────────────────────────────────────────
# HEADER PERSONALIZADO
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-header">
    <h1>🦟 Sistema de Predicción de Dengue</h1>
    <p>Departamento del Cesar, Colombia · Análisis histórico y predicciones 2026-2030</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis Histórico",
    "🩺 Evaluar Sintomas",
    "📈 Proyección Futura",
    "📋 Métricas del Modelo"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ANÁLISIS HISTÓRICO (GRÁFICOS PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title"> Análisis Histórico de Casos 2018-2025</div>', unsafe_allow_html=True)
    
    region_tab = st.radio("Región:", ["Cesar (Todo el departamento)", "Valledupar"], horizontal=True)
    archivo_anual = "historico_anual.csv" if "Cesar" in region_tab else "historico_anual_valledupar.csv"
    
    hist_anual = load_csv(archivo_anual)
    hist_edad = load_csv("historico_edad.csv")
    hist_estrato = load_csv("historico_estrato.csv")
    hist_sexo = load_csv("historico_sexo.csv")
    
    # KPIs estilo App B
    if not hist_anual.empty:
        total_casos = int(hist_anual["total"].sum())
        total_graves = int(hist_anual["graves"].sum()) if "graves" in hist_anual.columns else 0
        pct_graves = total_graves / total_casos * 100 if total_casos > 0 else 0
        
        colors = get_theme_colors()
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown(f"""
        <div style='background:linear-gradient(135deg,{colors["primary"]}22,{colors["primary"]}11);
                    border:2px solid {colors["primary"]}; border-radius:12px;
                    padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{colors["primary"]};'>{total_casos:,}</div>
            <div style='color:var(--text-secondary); font-size:0.85rem; margin-top:4px;'>Total Casos Históricos</div>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
        <div style='background:linear-gradient(135deg,{colors["danger"]}22,{colors["danger"]}11);
                    border:2px solid {colors["danger"]}; border-radius:12px;
                    padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{colors["danger"]};'>{total_graves:,}</div>
            <div style='color:var(--text-secondary); font-size:0.85rem; margin-top:4px;'>Casos Dengue Grave</div>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
        <div style='background:linear-gradient(135deg,{colors["warning"]}22,{colors["warning"]}11);
                    border:2px solid {colors["warning"]}; border-radius:12px;
                    padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{colors["warning"]};'>{pct_graves:.1f}%</div>
            <div style='color:var(--text-secondary); font-size:0.85rem; margin-top:4px;'>% Dengue Grave</div>
        </div>
        """, unsafe_allow_html=True)
        
        anio_pico = int(hist_anual.loc[hist_anual["total"].idxmax(), "anio"]) if "anio" in hist_anual.columns else "---"
        col4.markdown(f"""
        <div style='background:linear-gradient(135deg,{colors["success"]}22,{colors["success"]}11);
                    border:2px solid {colors["success"]}; border-radius:12px;
                    padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{colors["success"]};'>{anio_pico}</div>
            <div style='color:var(--text-secondary); font-size:0.85rem; margin-top:4px;'>Año con más Casos</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico 1 - Evolución anual 
    st.markdown('<div class="section-title">📅 Evolución Anual de Casos</div>', unsafe_allow_html=True)
    
    if not hist_anual.empty and "anio" in hist_anual.columns:
        fig_anual = go.Figure()
        
        fig_anual.add_trace(go.Scatter(
            x=hist_anual["anio"], y=hist_anual["total"],
            mode='lines+markers', name='Total Casos',
            line=dict(color=COLOR_TOTAL, width=2.5),
            marker=dict(size=8),
            hovertemplate="Año: %{x}<br>Total: %{y:,.0f}<extra></extra>"
        ))
        
        if "graves" in hist_anual.columns:
            fig_anual.add_trace(go.Scatter(
                x=hist_anual["anio"], y=hist_anual["graves"],
                mode='lines+markers', name='Dengue Grave',
                line=dict(color=COLOR_GRAVE, width=2.5, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate="Año: %{x}<br>Graves: %{y:,.0f}<extra></extra>"
            ))
        
        fig_anual.update_layout(
            title='Evolución Anual de Casos de Dengue',
            xaxis_title='Año',
            yaxis_title='Número de Casos',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            hovermode='x unified',
            height=450
        )
        fig_anual = configurar_grafica_tema(fig_anual, height=450)
        st.plotly_chart(fig_anual, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
    
    # Gráficos 2 y 3 - Edad y Estrato
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-title">👤 Casos por Grupo de Edad</div>', unsafe_allow_html=True)
        if not hist_edad.empty:
            col_sum = "sum" if "sum" in hist_edad.columns else hist_edad.columns[1]
            col_count = "count" if "count" in hist_edad.columns else hist_edad.columns[2]
            grupos_edad = hist_edad.iloc[:, 0].astype(str)
            
            fig_edad = go.Figure()
            fig_edad.add_trace(go.Bar(
                x=grupos_edad, y=hist_edad[col_count],
                name='Total', marker_color=COLOR_TOTAL,
                text=hist_edad[col_count].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                hovertemplate="Grupo: %{x}<br>Total: %{y:,.0f}<extra></extra>"
            ))
            fig_edad.add_trace(go.Bar(
                x=grupos_edad, y=hist_edad[col_sum],
                name='Graves', marker_color=COLOR_GRAVE,
                text=hist_edad[col_sum].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                hovertemplate="Grupo: %{x}<br>Graves: %{y:,.0f}<extra></extra>"
            ))
            fig_edad.update_layout(
                title='Distribución de Casos por Grupo de Edad',
                barmode='group',
                xaxis_title='Grupo de Edad',
                yaxis_title='Número de Casos',
                height=450
            )
            fig_edad = configurar_grafica_tema(fig_edad, height=450)
            st.plotly_chart(fig_edad, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
    
    with col_b:
        st.markdown('<div class="section-title">🏠 Casos por Estrato Socioeconómico</div>', unsafe_allow_html=True)
        if not hist_estrato.empty:
            col_sum = "sum" if "sum" in hist_estrato.columns else hist_estrato.columns[1]
            col_count = "count" if "count" in hist_estrato.columns else hist_estrato.columns[2]
            estratos = [f"E{int(e)}" for e in hist_estrato.iloc[:, 0]]
            
            fig_estrato = go.Figure()
            fig_estrato.add_trace(go.Bar(
                x=estratos, y=hist_estrato[col_count],
                name='Total', marker_color=COLOR_TOTAL,
                text=hist_estrato[col_count].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                hovertemplate="Estrato: %{x}<br>Total: %{y:,.0f}<extra></extra>"
            ))
            fig_estrato.add_trace(go.Bar(
                x=estratos, y=hist_estrato[col_sum],
                name='Graves', marker_color=COLOR_GRAVE,
                text=hist_estrato[col_sum].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                hovertemplate="Estrato: %{x}<br>Graves: %{y:,.0f}<extra></extra>"
            ))
            fig_estrato.update_layout(
                title='Distribución de Casos por Estrato',
                barmode='group',
                xaxis_title='Estrato',
                yaxis_title='Número de Casos',
                height=450
            )
            fig_estrato = configurar_grafica_tema(fig_estrato, height=450)
            st.plotly_chart(fig_estrato, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
    
    # Gráfico 4 - Por sexo
    st.markdown('<div class="section-title">⚧ Casos por Sexo</div>', unsafe_allow_html=True)
    if not hist_sexo.empty:
        col_sum = "sum" if "sum" in hist_sexo.columns else hist_sexo.columns[1]
        col_count = "count" if "count" in hist_sexo.columns else hist_sexo.columns[2]
        labels = ["Femenino", "Masculino"][:len(hist_sexo)]
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            fig_sexo_total = go.Figure(data=[go.Pie(
                labels=labels, values=hist_sexo[col_count],
                marker_colors=[COLOR_TOTAL, "#7B61FF"],
                hole=0.3,
                textinfo='label+percent',
                hovertemplate="%{label}<br>Casos: %{value:,.0f}<br>%{percent}<extra></extra>"
            )])
            fig_sexo_total.update_layout(title='Distribución Total', height=400)
            fig_sexo_total = configurar_grafica_tema(fig_sexo_total, height=400)
            st.plotly_chart(fig_sexo_total, use_container_width=True, config={'displayModeBar': True})
        
        with col_p2:
            fig_sexo_grave = go.Figure(data=[go.Pie(
                labels=labels, values=hist_sexo[col_sum],
                marker_colors=[COLOR_GRAVE, "#FF9F43"],
                hole=0.3,
                textinfo='label+percent',
                hovertemplate="%{label}<br>Graves: %{value:,.0f}<br>%{percent}<extra></extra>"
            )])
            fig_sexo_grave.update_layout(title='Distribución Graves', height=400)
            fig_sexo_grave = configurar_grafica_tema(fig_sexo_grave, height=400)
            st.plotly_chart(fig_sexo_grave, use_container_width=True, config={'displayModeBar': True})
    
    # Gráfico 5 - Porcentaje de graves por año
    if not hist_anual.empty and "anio" in hist_anual.columns and "graves" in hist_anual.columns:
        hist_anual["pct_graves_anual"] = (hist_anual["graves"] / hist_anual["total"] * 100).fillna(0)
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=hist_anual["anio"], y=hist_anual["pct_graves_anual"],
            mode='lines+markers+text',
            line=dict(color=COLOR_WARNING, width=2.5),
            marker=dict(size=10),
            text=hist_anual["pct_graves_anual"].round(1).astype(str) + '%',
            textposition='top center',
            name='% Dengue Grave',
            hovertemplate="Año: %{x}<br>% Graves: %{y:.1f}%<extra></extra>"
        ))
        fig_pct.update_layout(
            title='Porcentaje de Dengue Grave por Año',
            xaxis_title='Año',
            yaxis_title='% Dengue Grave',
            height=400
        )
        fig_pct = configurar_grafica_tema(fig_pct, height=400)
        st.plotly_chart(fig_pct, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PREDICCIÓN INDIVIDUAL

with tab2:
    st.markdown('<div class="section-title">🩺 Predictor de Dengue Grave</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📋 <strong>Instrucciones:</strong> Selecciona los síntomas y características del paciente.
    El modelo analizará la información y estimará la probabilidad de que el caso sea dengue grave.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        model, scaler, features = load_artifacts()
        artifacts_ok = True
    except Exception as e:
        st.error(f"⚠️ No se pudieron cargar los artefactos del modelo: {e}")
        st.info("Ejecuta el notebook de Colab primero y descarga modelo.pkl, scaler.pkl y features.pkl.")
        artifacts_ok = False
    
    if artifacts_ok:
        st.markdown("### 👤 Datos del Paciente")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            edad = st.number_input("Edad (años)", min_value=0, max_value=120, value=18)
        with col2:
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
        with col3:
            estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6], index=1)
        
        st.markdown("### 🌡️ Síntomas Clínicos")
        st.caption("Selecciona los síntomas presentes en el paciente")
        
        symptom_labels = {
            "fiebre": "🌡️ Fiebre",
            "cefalea": "🤕 Cefalea",
            "dolrretroo": "👁️ Dolor retro-ocular",
            "malgias": "💪 Mialgias",
            "artralgia": "🦴 Artralgias",
            "erupcionr": "🔴 Erupción cutánea",
            "dolor_abdo": "🫃 Dolor abdominal",
            "vomito": "🤢 Vómito",
            "diarrea": "🚽 Diarrea",
            "somnolenci": "😴 Somnolencia",
            "hepatomeg": "🫁 Hepatomegalia",
        }
        
        col_s1, col_s2 = st.columns(2)
        symptom_values = {}
        feat_list = [f for f in features if f in symptom_labels]
        mid = len(feat_list) // 2
        
        for i, feat in enumerate(feat_list):
            col = col_s1 if i < mid else col_s2
            with col:
                symptom_values[feat] = st.checkbox(symptom_labels[feat], value=False)
        
        st.markdown("---")
        threshold = st.slider("🎚️ Threshold de decisión", 0.2, 0.8, 0.4, 0.05,
                              help="Un threshold menor aumenta la sensibilidad (más recalls, más falsos positivos).")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predecir = st.button("🔍 Predecir Riesgo de Dengue Grave", type="primary", use_container_width=True)
        
        if predecir:
            row = {}
            for feat in features:
                if feat in symptom_labels:
                    row[feat] = int(symptom_values.get(feat, False))
                elif feat == "edad":
                    row[feat] = edad
                elif feat == "sexo_M" or feat == "sexo__M":
                    row[feat] = 1 if sexo == "Masculino" else 0
                elif feat == "estrato":
                    row[feat] = estrato
                else:
                    row[feat] = 0
            
            input_df = pd.DataFrame([row])
            input_df = input_df[features]
            
            with st.expander("🔧 Debug - Input al modelo"):
                st.write(input_df)
            
            input_sc = scaler.transform(input_df)
            prob = model.predict_proba(input_sc)[0][1]
            pred = int(prob >= threshold)
            
            st.markdown("---")
            st.markdown("## 📊 Resultado de la Evaluación")
            
            # Medidor tipo gauge con Plotly
            colors = get_theme_colors()
            color_gauge = COLOR_GRAVE if pred == 1 else "#28a745"
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Probabilidad de Dengue Grave (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color_gauge},
                    'steps': [
                        {'range': [0, 20], 'color': "#22c55e"},
                        {'range': [20, 50], 'color': "#f59e0b"},
                        {'range': [50, 80], 'color': "#ef4444"},
                        {'range': [80, 100], 'color': "#991b1b"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            fig_gauge = configurar_grafica_tema(fig_gauge, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': True})
            
            # Resultado textual
            if pred == 1:
                st.markdown(f"""
                <div class='danger-box'>
                    <b>⚠️ ALERTA: Riesgo de Dengue Grave</b><br>
                    Probabilidad estimada: <b>{prob:.1%}</b><br>
                    <i>Se recomienda hospitalización y monitoreo estrecho.</i>
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.25:
                st.markdown(f"""
                <div class='warning-box'>
                    <b>⚠️ Riesgo Moderado</b><br>
                    Probabilidad estimada: <b>{prob:.1%}</b><br>
                    <i>Monitorear evolución clínica. Considerar seguimiento.</i>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='info-box'>
                    <b>✅ Riesgo Bajo</b><br>
                    Probabilidad estimada: <b>{prob:.1%}</b><br>
                    <i>Sin indicadores de dengue grave actualmente.</i>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Síntomas reportados:**")
            activos = [symptom_labels[k] for k, v in symptom_values.items() if v and k in symptom_labels]
            if activos:
                for s in activos:
                    st.write(f"  ✅ {s}")
            else:
                st.write("  _Ninguno seleccionado_")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PROYECCIÓN FUTURA (GRÁFICOS PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📈 Proyección de Casos - Próximos 5 Años</div>', unsafe_allow_html=True)
    
    region_tab2 = st.radio("Región:", ["Cesar", "Valledupar"], horizontal=True)
    
    if region_tab2 == "Cesar":
        hist = load_csv("historico_anual.csv")
        proy = load_csv("predicciones_cesar.csv")
    else:
        hist = load_csv("historico_anual_valledupar.csv")
        proy = load_csv("predicciones_valledupar.csv")
    
    if hist.empty or proy.empty:
        st.info(" Ejecuta el notebook de Colab para generar los archivos de predicción.")
    else:
        total_proy = int(proy["total_casos"].sum()) if "total_casos" in proy.columns else 0
        graves_proy = int(proy["graves_proyectados"].sum()) if "graves_proyectados" in proy.columns else 0
        anio_max = int(proy["anio"].max()) if "anio" in proy.columns else "---"
        
        colors = get_theme_colors()
        col1, col2, col3 = st.columns(3)
        
        col1.markdown(f"""
        <div class="metric-card">
            <div>{anio_max}</div>
            <div>📅 Proyección hasta</div>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
        <div class="metric-card">
            <div>{total_proy:,}</div>
            <div>🔵 Casos totales proyectados</div>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
        <div class="metric-card">
            <div>{graves_proy:,}</div>
            <div>🔴 Graves proyectados</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico combinado histórico + proyección
        fig_proy = go.Figure()
        
        # Datos históricos
        fig_proy.add_trace(go.Scatter(
            x=hist["anio"], y=hist["total"],
            mode='lines+markers', name='Histórico Total',
            line=dict(color=COLOR_TOTAL, width=2.5),
            marker=dict(size=8),
            hovertemplate="Año: %{x}<br>Total: %{y:,.0f}<extra></extra>"
        ))
        
        if "graves" in hist.columns:
            fig_proy.add_trace(go.Scatter(
                x=hist["anio"], y=hist["graves"],
                mode='lines+markers', name='Histórico Graves',
                line=dict(color=COLOR_GRAVE, width=2.5, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate="Año: %{x}<br>Graves: %{y:,.0f}<extra></extra>"
            ))
        
        # Proyecciones
        if "total_casos" in proy.columns:
            fig_proy.add_trace(go.Scatter(
                x=proy["anio"], y=proy["total_casos"],
                mode='lines+markers', name='Proyección Total',
                line=dict(color=COLOR_PROY, width=2.5, dash='dot'),
                marker=dict(size=9),
                hovertemplate="Año: %{x}<br>Total: %{y:,.0f}<extra></extra>"
            ))
        
        if "graves_proyectados" in proy.columns:
            fig_proy.add_trace(go.Scatter(
                x=proy["anio"], y=proy["graves_proyectados"],
                mode='lines+markers', name='Proyección Graves',
                line=dict(color=COLOR_PROY_G, width=2.5, dash='dot'),
                marker=dict(size=9, symbol='diamond'),
                hovertemplate="Año: %{x}<br>Graves Proy: %{y:,.0f}<extra></extra>"
            ))
        
        # Línea vertical del presente
        fig_proy.add_vline(x=hist["anio"].max(), line_dash="dash", line_color="gray",
                           annotation_text="Hoy", annotation_position="top right")
        
        fig_proy.update_layout(
            title=f'Proyección de Casos de Dengue - {region_tab2}',
            xaxis_title='Año',
            yaxis_title='Número de Casos',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            hovermode='x unified',
            height=500
        )
        fig_proy = configurar_grafica_tema(fig_proy, height=500)
        st.plotly_chart(fig_proy, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
        
        # Tabla de proyección
        st.markdown("### 📋 Tabla de Proyecciones")
        proy_display = proy.drop(columns=["casos_proyectados"], errors="ignore")

        st.dataframe(proy_display.style.format({
            "total_casos": "{:,.0f}",
            "graves_proyectados": "{:,.0f}"
        }), use_container_width=True)
        
        st.caption("⚠️ Las proyecciones se basan en tendencia histórica de los últimos 3 años. No constituyen pronóstico epidemiológico oficial.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MÉTRICAS DEL MODELO (GRÁFICOS PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📋 Métricas del Modelo</div>', unsafe_allow_html=True)
    
    df_metrics = load_csv("metricas_modelos.csv")
    
    if df_metrics.empty:
        st.info("📂 Ejecuta el notebook de Colab para generar metricas_modelos.csv.")
    else:
        st.dataframe(
            df_metrics.style
                .background_gradient(subset=["recall", "f1", "roc_auc"], cmap="YlGn")
                .format({"recall": "{:.4f}", "precision": "{:.4f}",
                         "f1": "{:.4f}", "roc_auc": "{:.4f}", "pr_auc": "{:.4f}"}),
            use_container_width=True
        )
        
        st.markdown("### 📊 Comparación Visual")
        
        metricas_cols = ["recall", "precision", "f1", "roc_auc"]
        df_melt = df_metrics.melt(id_vars='modelo', value_vars=metricas_cols,
                                  var_name='Métrica', value_name='Valor')
        
        fig_met = px.bar(
            df_melt, x='Métrica', y='Valor', color='modelo', barmode='group',
            title='Comparación de Métricas por Modelo',
            color_discrete_sequence=[COLOR_TOTAL, COLOR_GRAVE, COLOR_PROY, "#7B61FF"],
            text=df_melt['Valor'].round(3).astype(str)
        )
        fig_met.update_traces(textposition='outside')
        fig_met.update_layout(
            yaxis_range=[0, 1.1],
            height=500,
            xaxis_title='Métrica',
            yaxis_title='Valor'
        )
        fig_met = configurar_grafica_tema(fig_met, height=500)
        st.plotly_chart(fig_met, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
        
        # Highlight mejor modelo por recall
        if 'recall' in df_metrics.columns:
            mejor = df_metrics.loc[df_metrics['recall'].idxmax(), 'modelo']
            st.success(f"🥇 **Mejor modelo por Recall (detección de casos graves):** {mejor}")
        
        st.markdown("""
        **Guía de métricas:**
        - **Recall** (prioridad): % de casos graves detectados correctamente. ¡No queremos falsos negativos!
        - **Precision**: De los que el modelo alerta como graves, cuántos realmente lo son.
        - **F1**: Balance entre Recall y Precision.
        - **ROC-AUC**: Capacidad discriminativa global del modelo.
        """)