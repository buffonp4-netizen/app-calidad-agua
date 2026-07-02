import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# =========================================================================
# 1. CONFIGURACIÓN E ID DE DRIVE
# =========================================================================
ID_GOOGLE_DRIVE = "1IVhxte4JME6DyFVZ7Dgq9OLMGcc5B21s"
MODEL_PATH = "modelo_agua.pkl"

st.set_page_config(page_title="Sistema Híbrido - Calidad del Agua", page_icon="💧", layout="wide")

# CSS para las alertas
st.markdown("""
<style>
    .card-potable { padding: 20px; background-color: #D1E7DD; border-radius: 10px; border-left: 8px solid #0F5132; color: #0F5132; }
    .card-nopotable { padding: 20px; background-color: #F8D7DA; border-radius: 10px; border-left: 8px solid #842029; color: #842029; }
    .treatment-box { padding: 15px; background-color: #FFF3CD; border-radius: 8px; border-left: 5px solid #FFC107; color: #664D03; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 2. CATÁLOGO UNIFICADO DE PARÁMETROS
# =========================================================================
# Incluye tanto los necesarios para el ML como los extras para normativa
CATALOGO_COMPLETO = {
    'pH': {'minsa': (6.5, 8.5), 'eca_a1': (6.5, 8.5), 'default': 7.0, 'is_ml': True},
    'CE': {'minsa': 1500.0, 'eca_a1': 1500.0, 'default': 400.0, 'is_ml': True},
    'T': {'minsa': 35.0, 'eca_a1': 25.0, 'default': 18.0, 'is_ml': True},
    'OD': {'minsa': 4.0, 'eca_a1': 6.0, 'default': 7.5, 'is_ml': True, 'invertido': True},
    'DBO': {'minsa': 3.0, 'eca_a1': 3.0, 'default': 2.0, 'is_ml': True},
    'CT': {'minsa': 0.0, 'eca_a1': 0.0, 'default': 10.0, 'is_ml': True},
    'AyG': {'minsa': 0.5, 'eca_a1': 0.5, 'default': 0.1, 'is_ml': True},
    'ArT': {'minsa': 0.01, 'eca_a1': 0.01, 'default': 0.002, 'is_ml': True},
    'PbT': {'minsa': 0.01, 'eca_a1': 0.01, 'default': 0.001, 'is_ml': True},
    'CuT': {'minsa': 2.0, 'eca_a1': 2.0, 'default': 0.005, 'is_ml': True},
    'MnT': {'minsa': 0.4, 'eca_a1': 0.4, 'default': 0.02, 'is_ml': True},
    'Ca': {'minsa': 200.0, 'eca_a1': 200.0, 'default': 45.0, 'is_ml': True},
    'Mg': {'minsa': 150.0, 'eca_a1': 150.0, 'default': 12.0, 'is_ml': True},
    'Dureza': {'minsa': 500.0, 'eca_a1': 500.0, 'default': 150.0, 'is_ml': True},
    # Extras
    'Cadmio (Total)': {'minsa': 0.003, 'eca_a1': 0.003, 'default': 0.001, 'is_ml': False},
    'Mercurio (Total)': {'minsa': 0.001, 'eca_a1': 0.001, 'default': 0.0005, 'is_ml': False},
    'Hierro (Total)': {'minsa': 0.3, 'eca_a1': 0.3, 'default': 0.1, 'is_ml': False},
    'Sulfatos': {'minsa': 250.0, 'eca_a1': 250.0, 'default': 80.0, 'is_ml': False},
    'Turbidez': {'minsa': 5.0, 'eca_a1': 5.0, 'default': 2.0, 'is_ml': False}
}

# Orden estricto que espera el modelo (los que tienen is_ml=True)
ORDER_ML = [k for k, v in CATALOGO_COMPLETO.items() if v['is_ml']]

# =========================================================================
# 3. CARGA DE MODELO
# =========================================================================
@st.cache_resource
def load_water_model():
    if not os.path.exists(MODEL_PATH):
        if ID_GOOGLE_DRIVE == "TU_ID_AQUI": return None
        url = f'https://drive.google.com/uc?id={ID_GOOGLE_DRIVE}'
        gdown.download(url, MODEL_PATH, quiet=False)
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

modelo_pipeline = load_water_model()

# =========================================================================
# 4. INTERFAZ
# =========================================================================
st.title("💧 Sistema Híbrido de Calidad del Agua")

# Captura de datos
seleccionados = st.multiselect("Seleccionar parámetros a evaluar:", list(CATALOGO_COMPLETO.keys()), default=ORDER_ML)
datos_ingresados = {}

cols = st.columns(3)
for i, param in enumerate(seleccionados):
    with cols[i % 3]:
        datos_ingresados[param] = st.number_input(param, value=CATALOGO_COMPLETO[param]['default'], format="%.4f")

if st.button("Evaluar y Predecir"):
    # 1. Evaluación Normativa
    st.subheader("Evaluación Legal (MINSA)")
    fallas = []
    for param, valor in datos_ingresados.items():
        lim = CATALOGO_COMPLETO[param]['minsa']
        if isinstance(lim, tuple): # Caso Rango
            if not (lim[0] <= valor <= lim[1]): fallas.append(param)
        else: # Caso Valor único
            if valor > lim: fallas.append(param)
    
    if not fallas:
        st.markdown('<div class="card-potable">✅ Cumple MINSA</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card-nopotable">❌ Fallas en: {", ".join(fallas)}</div>', unsafe_allow_html=True)

    # 2. Predicción IA
    st.subheader("Predicción IA")
    if modelo_pipeline:
        # Preparar vector para ML (solo los parámetros necesarios)
        vector_ml = pd.DataFrame([ [datos_ingresados.get(f, 0.0) for f in ORDER_ML] ], columns=ORDER_ML)
        prob = modelo_pipeline.predict_proba(vector_ml)[0][1]
        
        st.metric("Probabilidad de Potabilidad", f"{prob*100:.1f}%")
        st.progress(float(prob))
    else:
        st.warning("Modelo IA no disponible.")
