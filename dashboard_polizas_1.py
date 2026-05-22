"""
Dashboard Streamlit — Predicción de Renovación de Pólizas
Modelo: Random Forest con SMOTETomek y umbral óptimo por F1
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime
import io
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Pólizas",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Estilos CSS personalizados
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  
  /* Header principal */
  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0c4a6e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(56,189,248,0.2);
    box-shadow: 0 4px 32px rgba(0,0,0,0.3);
  }
  .main-header h1 {
    color: #f0f9ff;
    font-size: 2rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: #7dd3fc;
    font-size: 0.95rem;
    margin: 0;
    font-family: 'Space Mono', monospace;
  }

  /* Tarjeta de resultado */
  .result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 2px solid;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
  }
  .result-renovada {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border-color: #10b981;
  }
  .result-no-renovada {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
    border-color: #ef4444;
  }
  .result-title {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
  }
  .result-value {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0.25rem 0;
  }

  /* Badges de riesgo */
  .badge-alto   { background:#fef2f2; color:#b91c1c; border:1px solid #fca5a5; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }
  .badge-medio  { background:#fffbeb; color:#92400e; border:1px solid #fcd34d; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }
  .badge-bajo   { background:#f0fdf4; color:#14532d; border:1px solid #6ee7b7; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }

  /* Métricas */
  .metric-box {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #334155;
    text-align: center;
  }
  .metric-label { color:#94a3b8; font-size:0.78rem; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:1px; }
  .metric-value { color:#f1f5f9; font-size:1.8rem; font-weight:700; margin-top:4px; }

  /* Sección separadora */
  .section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 1.5rem 0;
  }

  /* Upload zone */
  .upload-info {
    background:#0f172a;
    border:1px dashed #334155;
    border-radius:12px;
    padding:1rem;
    color:#94a3b8;
    font-size:0.85rem;
    margin-top:0.5rem;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f172a;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stDateInput label,
  section[data-testid="stSidebar"] .stCheckbox label {
    color: #cbd5e1 !important;
    font-size: 0.82rem;
    font-weight: 500;
  }

  .stButton > button {
    background: linear-gradient(135deg,#0369a1,#0284c7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all .2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg,#0284c7,#38bdf8);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(56,189,248,0.3);
  }

  .warning-box {
    background:#1c1917;
    border-left:4px solid #f59e0b;
    border-radius:0 8px 8px 0;
    padding:0.8rem 1rem;
    color:#fbbf24;
    font-size:0.85rem;
    margin-top:0.5rem;
  }
  
  footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Valores de referencia del notebook (para demo sin modelo .pkl)
# ─────────────────────────────────────────────────────────────────────────────
UMBRAL_DEFAULT = 0.42   # umbral óptimo típico del entrenamiento

COMPAÑIAS    = ["FID","ACE","AXA","MAPFRE","HDI","CHUBB","SURA","LIBERTY","RSA","BCI"]
EVENTOS      = ["NORMAL","RENOVACION","ENDOSO","ANULACION","EMISION"]
RAMOS        = ["Vehic- Motoriz. Pesados","Automóviles","Incendio","Vida","RC Civil",
                "Accidentes Personales","Hogar","Transporte","SOAT","Multi-riesgo"]
CANALES      = ["IMP","DIR","COR","BAN","AGN","DIG","RET"]
CIUDADES     = ["Santiago","Viña Del Mar","Concepción","Valparaíso","Antofagasta",
                "La Serena","Temuco","Rancagua","Puerto Montt","Iquique","Arica",
                "Talca","Chillán","Calama","Copiapó","Osorno","Coquimbo"]
LINEAS       = ["Old Gallagher","Corporativo","Pyme","Masivo","Agrobusiness","Marine","Aviation"]
SUCURSALES   = ["AJG Santiago Centro","AJG Viña del Mar","AJG Concepción","AJG Antofagasta",
                "AJG La Serena","AJG Temuco","AJG Puerto Montt","AJG Rancagua"]


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelo (si existe el .pkl)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo(uploaded_file):
    """Carga el bundle del modelo desde un archivo subido."""
    try:
        bundle = joblib.load(uploaded_file)
        return bundle, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento (replica exacta del notebook)
# ─────────────────────────────────────────────────────────────────────────────
def preprocesar_poliza(datos: dict, bundle: dict) -> np.ndarray:
    modelo      = bundle['modelo']
    scaler      = bundle['scaler']
    le_ciudad   = bundle['le_ciudad']
    COLUMNAS    = bundle['columnas']
    OHE_COLS    = bundle['ohe_cols']

    f_aviso = pd.to_datetime(datos['Fecha_Aviso'])
    ciudad  = datos['Ciudad_y']

    fila = {
        'Prima_Neta':              datos['Prima_Neta'],
        'Valor_Cuota':             datos['Valor_Cuota'],
        'Dias_Entre_Aviso_Cierre': datos['Dias_Entre_Aviso_Cierre'],
        'Riesgo_Pais':             datos['Riesgo_Pais'],
        'Sin_Plan_Pago':           int(datos['Sin_Plan_Pago']),
        'Garantia_Suscripcion':    int(datos['Garantia_Suscripcion']),
        'Mes_Aviso':               f_aviso.month,
        'DiaSemana_Aviso':         f_aviso.dayofweek,
    }

    fila['Ciudad_y'] = (
        int(le_ciudad.transform([ciudad])[0])
        if ciudad in le_ciudad.classes_ else 0
    )

    row = pd.DataFrame([fila])
    for col in COLUMNAS:
        if col not in row.columns:
            row[col] = 0

    mapa_campos = {
        'Canal':         datos['Canal'],
        'Evento':        datos['Evento'],
        'Sucursal':      datos['Sucursal'],
        'Compania':      datos['Compania'],
        'Linea_Negocio': datos['Linea_Negocio'],
        'Nombre_Ramo':   datos['Nombre_Ramo'],
    }
    for prefijo, valor in mapa_campos.items():
        col_ohe = f"{prefijo}_{valor}"
        if col_ohe in COLUMNAS:
            row[col_ohe] = 1

    row = row[COLUMNAS]
    return scaler.transform(row), modelo, bundle['umbral_optimo']


# ─────────────────────────────────────────────────────────────────────────────
# Predicción demo (sin modelo real cargado)
# ─────────────────────────────────────────────────────────────────────────────
def predecir_demo(datos: dict) -> dict:
    """Simula una predicción para demostración visual."""
    np.random.seed(
        int(datos['Prima_Neta'] * 7 +
            datos['Dias_Entre_Aviso_Cierre'] * 3 +
            datos['Riesgo_Pais'] * 100) % 2**31
    )
    # Heurística simple para demo
    score  = 0.0
    score += 0.3 if datos['Dias_Entre_Aviso_Cierre'] > 45 else -0.1
    score += 0.2 if datos['Riesgo_Pais'] > 1.5 else 0.0
    score += 0.15 if datos['Sin_Plan_Pago'] else -0.05
    score += 0.1 if datos['Garantia_Suscripcion'] else 0.0
    score += np.random.normal(0, 0.15)
    prob_no_renueva = float(np.clip(0.25 + score, 0.05, 0.95))
    prob_renueva    = 1 - prob_no_renueva
    umbral          = UMBRAL_DEFAULT
    prediccion      = int(prob_no_renueva >= umbral)
    riesgo = "Alto" if prob_no_renueva >= 0.7 else ("Medio" if prob_no_renueva >= 0.4 else "Bajo")
    return {
        'no_renovada':      prediccion,
        'resultado':        'No Renovada' if prediccion else 'Renovada',
        'riesgo':           riesgo,
        'prob_no_renovada': round(prob_no_renueva * 100, 2),
        'prob_renovada':    round(prob_renueva * 100, 2),
        'umbral_usado':     round(umbral, 3),
        'ciudad_conocida':  True,
        'modo':             'demo',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────────────────────────
def gauge_probabilidad(prob_no_renueva: float, umbral: float) -> go.Figure:
    color_aguja = "#ef4444" if prob_no_renueva >= umbral else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_no_renueva,
        delta={'reference': umbral * 100, 'valueformat': '.1f',
               'increasing': {'color': '#ef4444'}, 'decreasing': {'color': '#10b981'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': '#f1f5f9', 'family': 'DM Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#475569',
                     'tickfont': {'color': '#94a3b8', 'size': 11}},
            'bar':  {'color': color_aguja, 'thickness': 0.25},
            'bgcolor': '#1e293b',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],           'color': 'rgba(16,185,129,0.15)'},
                {'range': [40, 70],          'color': 'rgba(245,158,11,0.15)'},
                {'range': [70, 100],         'color': 'rgba(239,68,68,0.15)'},
            ],
            'threshold': {
                'line': {'color': '#f59e0b', 'width': 3},
                'thickness': 0.8,
                'value': umbral * 100,
            },
        },
        title={'text': "Probabilidad No Renovación",
               'font': {'size': 13, 'color': '#94a3b8', 'family': 'Space Mono'}},
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=10, l=30, r=30),
        font={'family': 'DM Sans'},
    )
    return fig


def barras_probabilidad(prob_no: float, prob_si: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Renovada', 'No Renovada'],
        y=[prob_si, prob_no],
        marker_color=['#10b981', '#ef4444'],
        marker_line_width=0,
        text=[f'{prob_si:.1f}%', f'{prob_no:.1f}%'],
        textposition='outside',
        textfont={'color': '#f1f5f9', 'size': 14, 'family': 'DM Sans'},
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 105], gridcolor='#1e293b', tickcolor='#475569',
                   tickfont={'color': '#94a3b8', 'size': 11}, ticksuffix='%'),
        xaxis=dict(tickfont={'color': '#cbd5e1', 'size': 13}),
        showlegend=False,
        margin=dict(t=20, b=10, l=10, r=10),
        font={'family': 'DM Sans'},
        bargap=0.45,
    )
    return fig


def radar_factores(datos: dict) -> go.Figure:
    """Radar chart con factores normalizados de riesgo."""
    categorias = ['Prima Alta', 'Días Aviso', 'Riesgo País', 'Sin Plan Pago', 'Garantía']
    valores = [
        min(datos['Prima_Neta'] / 500, 1.0),
        min(datos['Dias_Entre_Aviso_Cierre'] / 90, 1.0),
        min((datos['Riesgo_Pais'] - 0.8) / 1.5, 1.0),
        1.0 if datos['Sin_Plan_Pago'] else 0.0,
        1.0 if datos['Garantia_Suscripcion'] else 0.0,
    ]
    valores = [max(0, v) for v in valores]
    categorias_cierre = categorias + [categorias[0]]
    valores_cierre    = valores + [valores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores_cierre,
        theta=categorias_cierre,
        fill='toself',
        fillcolor='rgba(239,68,68,0.2)',
        line=dict(color='#ef4444', width=2),
        name='Factores de riesgo',
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(30,41,59,0.5)',
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont={'color': '#64748b', 'size': 9},
                            gridcolor='#334155', linecolor='#334155'),
            angularaxis=dict(tickfont={'color': '#cbd5e1', 'size': 11},
                             gridcolor='#334155', linecolor='#334155'),
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        font={'family': 'DM Sans'},
    )
    return fig


def tabla_lote(resultados: list[dict], datos_list: list[dict]) -> pd.DataFrame:
    rows = []
    for i, (res, dat) in enumerate(zip(resultados, datos_list)):
        rows.append({
            '#': i + 1,
            'Compañía':       dat.get('Compania', '—'),
            'Ramo':           dat.get('Nombre_Ramo', '—'),
            'Ciudad':         dat.get('Ciudad_y', '—'),
            'Prob. No Ren. %': res['prob_no_renovada'],
            'Resultado':      res['resultado'],
            'Riesgo':         res['riesgo'],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Carga del modelo y modo
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Predictor Pólizas")
    st.markdown("---")

    st.markdown("**Modelo Random Forest**")
    model_file = st.file_uploader(
        "Subir modelo (.pkl)",
        type=['pkl'],
        help="Sube el archivo polizas_rf_model.pkl generado en el notebook"
    )
    st.markdown(
        '<div class="upload-info">ℹ️ Sin modelo se usa modo demo con heurística visual</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    modo = st.radio("Modo de predicción", ["Póliza individual", "Lote (CSV)"], index=0)
    st.markdown("---")
    st.markdown("**Variables del modelo**")
    st.caption("Random Forest · 300 árboles · SMOTETomek · Umbral óptimo F1")

# ─────────────────────────────────────────────────────────────────────────────
# Cargar bundle si se subió
# ─────────────────────────────────────────────────────────────────────────────
bundle = None
if model_file is not None:
    bundle, err = cargar_modelo(model_file)
    if err:
        st.sidebar.error(f"Error cargando modelo: {err}")
        bundle = None
    else:
        st.sidebar.success("✅ Modelo cargado correctamente")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1> Predictor de Renovación de Pólizas</h1>
  <p>Random Forest · SMOTETomek · Umbral óptimo por F1-Score</p>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.info("⚡ **Modo demo activo** — Sube el archivo `polizas_rf_model.pkl` en la barra lateral para usar el modelo real.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# MODO: PÓLIZA INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
if modo == "Póliza individual":

    st.markdown("### 📋 Datos de la Póliza")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Información financiera**")
        prima     = st.number_input("Prima Bruta CLP", min_value=0.0, value=116.204, step=10.0, format="%.3f")
        cuota     = st.number_input("Valor Cuota", min_value=0.0, value=10.0, step=1.0)
        dias      = st.number_input("Días entre Aviso y Cierre", min_value=0, value=30, step=1)
        riesgo_p  = st.number_input("Riesgo País", min_value=0.5, max_value=5.0, value=1.17114, step=0.01, format="%.5f")

    with col2:
        st.markdown("**Clasificación**")
        compania  = st.selectbox("Compañía", COMPAÑIAS, index=0)
        evento    = st.selectbox("Evento", EVENTOS, index=0)
        ramo      = st.selectbox("Nombre Ramo", RAMOS, index=0)
        canal     = st.selectbox("Canal", CANALES, index=0)

    with col3:
        st.markdown("**Ubicación y operación**")
        ciudad      = st.selectbox("Ciudad", CIUDADES, index=0)
        linea       = st.selectbox("Línea de Negocio", LINEAS, index=0)
        sucursal    = st.selectbox("Sucursal", SUCURSALES, index=0)
        fecha_aviso = st.date_input("Fecha de Aviso", value=date(2024, 3, 15))
        sin_plan    = st.checkbox("Sin Plan de Pago")
        garantia    = st.checkbox("Garantía de Suscripción")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button("Predecir Renovación", use_container_width=True):
        datos = {
            'Prima_Neta':   prima,
            'Valor_Cuota':             cuota,
            'Dias_Entre_Aviso_Cierre': dias,
            'Riesgo_Pais':             riesgo_p,
            'Fecha_Aviso':             str(fecha_aviso),
            'Sin_Plan_Pago':           sin_plan,
            'Garantia_Suscripcion':    garantia,
            'Compania':                compania,
            'Evento':                  evento,
            'Nombre_Ramo':             ramo,
            'Canal':                   canal,
            'Ciudad_y':                ciudad,
            'Linea_Negocio':           linea,
            'Sucursal':                sucursal,
        }

        with st.spinner("Analizando póliza..."):
            if bundle:
                try:
                    row_scaled, modelo_rf, umbral = preprocesar_poliza(datos, bundle)
                    proba            = modelo_rf.predict_proba(row_scaled)[0]
                    prob_no_renueva  = float(proba[1])
                    prob_renueva     = float(proba[0])
                    prediccion       = int(prob_no_renueva >= umbral)
                    riesgo = "Alto" if prob_no_renueva >= 0.7 else ("Medio" if prob_no_renueva >= 0.4 else "Bajo")
                    resultado = {
                        'no_renovada':      prediccion,
                        'resultado':        'No Renovada' if prediccion else 'Renovada',
                        'riesgo':           riesgo,
                        'prob_no_renovada': round(prob_no_renueva * 100, 2),
                        'prob_renovada':    round(prob_renueva * 100, 2),
                        'umbral_usado':     round(umbral, 3),
                        'ciudad_conocida':  ciudad in bundle['le_ciudad'].classes_,
                        'modo':             'real',
                    }
                except Exception as e:
                    st.error(f"Error en predicción: {e}")
                    resultado = None
            else:
                resultado = predecir_demo(datos)

        if resultado:
            st.markdown("---")
            st.markdown("### 📊 Resultado del Análisis")

            # ── Tarjeta principal + gauge
            r1, r2 = st.columns([1, 2])
            with r1:
                css_class = "result-no-renovada" if resultado['no_renovada'] else "result-renovada"
                emoji     = "⚠️" if resultado['no_renovada'] else "✅"
                color_val = "#ef4444" if resultado['no_renovada'] else "#10b981"
                riesgo_badge = {
                    'Alto':  '<span class="badge-alto">🔴 Riesgo Alto</span>',
                    'Medio': '<span class="badge-medio">🟡 Riesgo Medio</span>',
                    'Bajo':  '<span class="badge-bajo">🟢 Riesgo Bajo</span>',
                }[resultado['riesgo']]

                st.markdown(f"""
                <div class="result-card {css_class}">
                  <div class="result-title">Predicción</div>
                  <div class="result-value" style="color:{color_val}">{emoji} {resultado['resultado']}</div>
                  <br>{riesgo_badge}<br><br>
                  <div style="color:rgba(255,255,255,0.5);font-size:0.8rem;font-family:'Space Mono',monospace;">
                    Umbral: {resultado['umbral_usado']}
                    {'· ⚡ Demo' if resultado['modo']=='demo' else '· 🤖 Modelo Real'}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if not resultado.get('ciudad_conocida', True):
                    st.markdown('<div class="warning-box">⚠️ Ciudad no vista en entrenamiento — se usó codificación por defecto</div>', unsafe_allow_html=True)

            with r2:
                st.plotly_chart(
                    gauge_probabilidad(resultado['prob_no_renovada'] / 100, resultado['umbral_usado']),
                    use_container_width=True,
                    config={'displayModeBar': False},
                )

            # ── Métricas + barras + radar
            st.markdown("#### Detalle probabilístico")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-box">
                  <div class="metric-label">Prob. No Renovada</div>
                  <div class="metric-value" style="color:#ef4444">{resultado['prob_no_renovada']}%</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box">
                  <div class="metric-label">Prob. Renovada</div>
                  <div class="metric-value" style="color:#10b981">{resultado['prob_renovada']}%</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box">
                  <div class="metric-label">Umbral Óptimo</div>
                  <div class="metric-value" style="color:#f59e0b">{resultado['umbral_usado']}</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                margen = abs(resultado['prob_no_renovada']/100 - resultado['umbral_usado'])
                st.markdown(f"""<div class="metric-box">
                  <div class="metric-label">Margen al umbral</div>
                  <div class="metric-value" style="color:#7dd3fc">{margen:.3f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Distribución de probabilidades**")
                st.plotly_chart(
                    barras_probabilidad(resultado['prob_no_renovada'], resultado['prob_renovada']),
                    use_container_width=True,
                    config={'displayModeBar': False},
                )
            with g2:
                st.markdown("**Factores de riesgo detectados**")
                st.plotly_chart(
                    radar_factores(datos),
                    use_container_width=True,
                    config={'displayModeBar': False},
                )

            # ── Resumen de entrada
            with st.expander("📄 Ver datos ingresados"):
                df_resumen = pd.DataFrame([{
                    'Variable': k, 'Valor': str(v)
                } for k, v in datos.items()])
                st.dataframe(df_resumen, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODO: LOTE (CSV)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("### 📂 Predicción por Lote")

    # Plantilla CSV
    plantilla_cols = [
        'Prima_Neta','Valor_Cuota','Dias_Entre_Aviso_Cierre','Riesgo_Pais',
        'Fecha_Aviso','Sin_Plan_Pago','Garantia_Suscripcion',
        'Compania','Evento','Nombre_Ramo','Canal','Ciudad_y','Linea_Negocio','Sucursal'
    ]
    df_plantilla = pd.DataFrame([{
        'Prima_Neta':              116.204,
        'Valor_Cuota':             10.0,
        'Dias_Entre_Aviso_Cierre': 30.0,
        'Riesgo_Pais':             1.17114,
        'Fecha_Aviso':             '2024-03-15',
        'Sin_Plan_Pago':           False,
        'Garantia_Suscripcion':    False,
        'Compania':                'FID',
        'Evento':                  'NORMAL',
        'Nombre_Ramo':             'Vehic- Motoriz. Pesados',
        'Canal':                   'IMP',
        'Ciudad_y':                'Viña Del Mar',
        'Linea_Negocio':           'Old Gallagher',
        'Sucursal':                'AJG Viña del Mar',
    }])

    csv_bytes = df_plantilla.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar plantilla CSV",
        data=csv_bytes,
        file_name="plantilla_polizas.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader("📤 Subir CSV con pólizas", type=['csv'])

    if uploaded_csv:
        df_lote = pd.read_csv(uploaded_csv)
        st.success(f"✅ {len(df_lote)} pólizas cargadas")
        st.dataframe(df_lote.head(), use_container_width=True)

        if st.button("🔍  Predecir lote completo", use_container_width=True):
            resultados  = []
            datos_list  = []

            progress = st.progress(0, text="Procesando pólizas...")

            for i, row in df_lote.iterrows():
                datos = {
                    'Prima_Neta':   float(row.get('Prima_Neta', 100)),
                    'Valor_Cuota':             float(row.get('Valor_Cuota', 10)),
                    'Dias_Entre_Aviso_Cierre': float(row.get('Dias_Entre_Aviso_Cierre', 30)),
                    'Riesgo_Pais':             float(row.get('Riesgo_Pais', 1.2)),
                    'Fecha_Aviso':             str(row.get('Fecha_Aviso', '2024-01-01')),
                    'Sin_Plan_Pago':           bool(row.get('Sin_Plan_Pago', False)),
                    'Garantia_Suscripcion':    bool(row.get('Garantia_Suscripcion', False)),
                    'Compania':                str(row.get('Compania', 'FID')),
                    'Evento':                  str(row.get('Evento', 'NORMAL')),
                    'Nombre_Ramo':             str(row.get('Nombre_Ramo', 'Automóviles')),
                    'Canal':                   str(row.get('Canal', 'DIR')),
                    'Ciudad_y':                str(row.get('Ciudad_y', 'Santiago')),
                    'Linea_Negocio':           str(row.get('Linea_Negocio', 'Masivo')),
                    'Sucursal':                str(row.get('Sucursal', 'AJG Santiago Centro')),
                }
                if bundle:
                    try:
                        row_scaled, modelo_rf, umbral = preprocesar_poliza(datos, bundle)
                        proba           = modelo_rf.predict_proba(row_scaled)[0]
                        prob_no         = float(proba[1])
                        prediccion      = int(prob_no >= umbral)
                        riesgo = "Alto" if prob_no >= 0.7 else ("Medio" if prob_no >= 0.4 else "Bajo")
                        resultados.append({'resultado': 'No Renovada' if prediccion else 'Renovada',
                                           'prob_no_renovada': round(prob_no * 100, 2),
                                           'riesgo': riesgo})
                    except:
                        resultados.append({'resultado': 'Error', 'prob_no_renovada': 0, 'riesgo': '—'})
                else:
                    resultados.append(predecir_demo(datos))
                datos_list.append(datos)
                progress.progress((i + 1) / len(df_lote), text=f"Procesando {i+1}/{len(df_lote)}...")

            progress.empty()

            df_res = tabla_lote(resultados, datos_list)
            st.markdown("---")
            st.markdown("### 📊 Resultados del Lote")

            # KPIs rápidos
            total      = len(df_res)
            no_renov   = int((df_res['Resultado'] == 'No Renovada').sum())
            alto_riesgo= int((df_res['Riesgo'] == 'Alto').sum())

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class="metric-box"><div class="metric-label">Total Pólizas</div>
                <div class="metric-value">{total}</div></div>""", unsafe_allow_html=True)
            with k2:
                pct = round(no_renov/total*100, 1)
                st.markdown(f"""<div class="metric-box"><div class="metric-label">No Renovadas</div>
                <div class="metric-value" style="color:#ef4444">{no_renov} ({pct}%)</div></div>""", unsafe_allow_html=True)
            with k3:
                pct2 = round((total-no_renov)/total*100, 1)
                st.markdown(f"""<div class="metric-box"><div class="metric-label">Renovadas</div>
                <div class="metric-value" style="color:#10b981">{total-no_renov} ({pct2}%)</div></div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""<div class="metric-box"><div class="metric-label">Riesgo Alto</div>
                <div class="metric-value" style="color:#f59e0b">{alto_riesgo}</div></div>""", unsafe_allow_html=True)

            st.markdown("")

            # Gráfico distribución
            g1, g2 = st.columns(2)
            with g1:
                conteo = df_res['Resultado'].value_counts().reset_index()
                fig_pie = px.pie(
                    conteo, values='count', names='Resultado',
                    color='Resultado',
                    color_discrete_map={'Renovada': '#10b981', 'No Renovada': '#ef4444'},
                    title='Distribución de resultados',
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#f1f5f9', 'family': 'DM Sans'},
                    height=300,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

            with g2:
                probs = [r['prob_no_renovada'] for r in resultados]
                fig_hist = go.Figure(go.Histogram(
                    x=probs, nbinsx=15,
                    marker_color='#3b82f6', marker_line_color='#1e3a5f', marker_line_width=1,
                    name='Distribución',
                ))
                fig_hist.add_vline(
                    x=UMBRAL_DEFAULT * 100, line_dash='dash', line_color='#f59e0b',
                    annotation_text='Umbral', annotation_font_color='#f59e0b',
                )
                fig_hist.update_layout(
                    title='Distribución de probabilidades',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#f1f5f9', 'family': 'DM Sans'},
                    height=300,
                    xaxis=dict(title='Prob. No Renovada (%)', gridcolor='#1e293b', tickcolor='#475569'),
                    yaxis=dict(gridcolor='#1e293b', tickcolor='#475569'),
                    margin=dict(t=40, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

            # Tabla resultados
            st.markdown("**Detalle por póliza**")

            def color_resultado(val):
                if val == 'No Renovada':
                    return 'color: #ef4444; font-weight: 600'
                elif val == 'Renovada':
                    return 'color: #10b981; font-weight: 600'
                return ''

            def color_riesgo(val):
                mapa = {'Alto': 'color: #ef4444', 'Medio': 'color: #f59e0b', 'Bajo': 'color: #10b981'}
                return mapa.get(val, '')

            styled = df_res.style.map(color_resultado, subset=['Resultado']) \
                                  .map(color_riesgo, subset=['Riesgo'])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Descarga
            csv_out = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar resultados CSV",
                data=csv_out,
                file_name=f"predicciones_polizas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )
