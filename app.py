import streamlit as st
import pandas as pd
import os
import joblib
import gdown

# --- CONFIGURACIÓN ---
ID_GOOGLE_DRIVE = "TU_ID_AQUI" 
MODEL_PATH = "modelo_agua.pkl"

# --- TODO UNIFICADO EN UN SOLO DICCIONARIO ---
# He combinado los base y los extras para que no se pierda nada
CATALOGO_PARAMETROS = {
    'pH': {'minsa': (6.5, 8.5), 'default': 7.0},
    'CE': {'minsa': 1500.0, 'default': 400.0},
    'T': {'minsa': 35.0, 'default': 20.0},
    'OD': {'minsa': 4.0, 'default': 7.0},
    'DBO': {'minsa': 3.0, 'default': 2.0},
    'CT': {'minsa': 0.0, 'default': 0.0},
    'AyG': {'minsa': 0.5, 'default': 0.1},
    'ArT': {'minsa': 0.01, 'default': 0.002},
    'PbT': {'minsa': 0.01, 'default': 0.001},
    'CuT': {'minsa': 2.0, 'default': 0.005},
    'MnT': {'minsa': 0.4, 'default': 0.02},
    'Ca': {'minsa': 200.0, 'default': 50.0},
    'Mg': {'minsa': 150.0, 'default': 20.0},
    'Dureza': {'minsa': 500.0, 'default': 100.0},
    # Extras
    'Cadmio (Total)': {'minsa': 0.003, 'default': 0.001},
    'Mercurio (Total)': {'minsa': 0.001, 'default': 0.0005},
    'Hierro (Total)': {'minsa': 0.3, 'default': 0.1},
    'Sulfatos': {'minsa': 250.0, 'default': 50.0},
    'Turbidez': {'minsa': 5.0, 'default': 1.0},
    'Nitratos': {'minsa': 50.0, 'default': 10.0},
    'Nitritos': {'minsa': 3.0, 'default': 0.1},
    'Cloruros': {'minsa': 250.0, 'default': 50.0},
    'Fluoruros': {'minsa': 1.5, 'default': 0.5}
}

# --- LÓGICA DE CARGA ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        if ID_GOOGLE_DRIVE == "TU_ID_AQUI":
            return None
        url = f'https://drive.google.com/uc?id={ID_GOOGLE_DRIVE}'
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

modelo = load_model()

# --- INTERFAZ ---
st.title("💧 Sistema de Monitoreo de Agua")

st.subheader("Seleccionar parámetros para evaluación")

# Usamos el catálogo completo para el multiselect
seleccionados = st.multiselect(
    "Elige todos los parámetros que deseas evaluar:", 
    options=list(CATALOGO_PARAMETROS.keys())
)

# Crear los inputs dinámicamente según lo que el usuario seleccionó
datos_usuario = {}
if seleccionados:
    cols = st.columns(3)
    for i, param in enumerate(seleccionados):
        with cols[i % 3]:
            datos_usuario[param] = st.number_input(
                f"{param}", 
                value=CATALOGO_PARAMETROS[param]['default'],
                format="%.4f"
            )

if st.button("Analizar"):
    if not datos_usuario:
        st.warning("Selecciona al menos un parámetro.")
    else:
        st.write("### Resultados")
        for param, valor in datos_usuario.items():
            limite = CATALOGO_PARAMETROS[param]['minsa']
            # Lógica simple
            if isinstance(limite, tuple):
                if limite[0] <= valor <= limite[1]:
                    st.success(f"{param}: OK ({valor})")
                else:
                    st.error(f"{param}: FUERA DE RANGO ({valor})")
            else:
                if valor <= limite:
                    st.success(f"{param}: OK ({valor})")
                else:
                    st.error(f"{param}: SUPERA LÍMITE ({valor})")
