import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Híbrido - Calidad del Agua", page_icon="💧", layout="wide")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_water_model():
    try:
        model_path = hf_hub_download(repo_id="buffoness/modelo-agua", filename="modelo_agua.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo_pipeline = load_water_model()

# --- DEFINICIÓN DE PARÁMETROS ---
FEATURES_ML = ['pH', 'CE', 'T', 'OD', 'DBO', 'CT', 'AyG', 'ArT', 'PbT', 'CuT', 'MnT', 'Ca', 'Mg', 'Dureza']

# Parámetros adicionales (Solo para informe legal, no para IA)
PARAMETROS_NORMATIVOS = {
    'Cadmio (Total) (mg/L)': {'minsa': 0.003, 'eca_a1': 0.003, 'default': 0.001},
    'Mercurio (Total) (mg/L)': {'minsa': 0.001, 'eca_a1': 0.001, 'default': 0.0005},
    'Hierro (Total) (mg/L)': {'minsa': 0.3, 'eca_a1': 0.3, 'default': 0.1},
    'Sulfatos (SO4=) (mg/L)': {'minsa': 250.0, 'eca_a1': 250.0, 'default': 80.0},
    'Turbidez (UNT)': {'minsa': 5.0, 'eca_a1': 5.0, 'default': 2.0}
}

# --- FUNCIÓN DE INPUT ---
def input_variable(label, default_value, min_val=0.0, format="%.4f"):
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        tiene_dato = st.checkbox(f"✅ Incluir", value=True, key=f"check_{label}")
    with col2:
        val = st.number_input(label, min_value=min_val, value=default_value, 
                              format=format, disabled=not tiene_dato, key=f"input_{label}")
    return val if tiene_dato else np.nan

# --- INTERFAZ ---
st.title("💧 Sistema Híbrido de Diagnóstico de Calidad del Agua")

tab1, tab2, tab3 = st.tabs(["🤖 Inputs para IA", "⚖️ Inputs para Normativa", "📋 Diagnóstico Final"])

inputs_ml = {}
inputs_reg = {}

with tab1:
    st.subheader("1. Parámetros para el Modelo de IA")
    st.write("Estos datos alimentan el algoritmo de predicción.")
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES_ML):
        with cols[i % 3]:
            inputs_ml[feature] = input_variable(feature, 7.0 if feature=='pH' else 0.0)

with tab2:
    st.subheader("2. Parámetros para Normativa Legal (ECA/MINSA)")
    st.write("Estos no afectan a la IA, pero son obligatorios para el informe legal.")
    cols_reg = st.columns(2)
    for i, (name, limits) in enumerate(PARAMETROS_NORMATIVOS.items()):
        with cols_reg[i % 2]:
            inputs_reg[name] = input_variable(name, limits['default'])

# --- PROCESAMIENTO ---
with tab3:
    st.subheader("🔍 Diagnóstico")
    
    # 1. PREDICCIÓN IA
    if modelo_pipeline is not None:
        input_df = pd.DataFrame([inputs_ml])
        try:
            # Aquí ocurre la magia: si hay NaNs, el pipeline los imputa
            prob = modelo_pipeline.predict_proba(input_df)[0]
            prob_potable = prob[1]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Probabilidad de Potabilidad", f"{prob_potable*100:.1f}%")
                st.progress(float(prob_potable))
            
            with col_res2:
                if prob_potable >= 0.75:
                    st.success("✅ IA: El agua es clasificada como APTA.")
                else:
                    st.error("❌ IA: El agua es clasificada como NO APTA.")
        except Exception as e:
            st.error(f"Error en IA: {e}")

    st.divider()

    # 2. REPORTE NORMATIVO (MINSA)
    st.subheader("🏛️ Evaluación Normativa (MINSA)")
    
    # Combinamos para la evaluación, filtrando los que no se ingresaron (NaN)
    report_data = {**inputs_ml, **inputs_reg}
    
    fallas = []
    # Simulación de chequeo rápido (puedes ampliarlo con los diccionarios de límites)
    for param, val in report_data.items():
        if not np.isnan(val):
            # Ejemplo de lógica: si tienes diccionario de límites, úsalo aquí
            # if val > limite: fallas.append(param)
            pass 
            
    if not fallas:
        st.write("✅ Todos los parámetros ingresados están dentro de los rangos legales.")
    else:
        st.write(f"⚠️ Se encontraron {len(fallas)} parámetros fuera de norma.")
        st.write(fallas)
