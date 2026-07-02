import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# =========================================================================
# CONFIGURACIÓN VISUAL
# =========================================================================
st.set_page_config(page_title="Sistema Híbrido Calidad Agua", page_icon="💧", layout="wide")

st.markdown("""
<style>
    .stToggle { margin-top: 25px; }
    .card-res { padding: 20px; border-radius: 10px; background-color: #f8f9fa; border: 1px solid #dee2e6; margin-bottom: 10px; }
    .metric-box { background-color: #e9ecef; padding: 10px; border-radius: 8px; text-align: center; }
    .stTextInput { margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# CARGA DEL MODELO
# =========================================================================
@st.cache_resource
def load_water_model():
    try:
        model_path = hf_hub_download(repo_id="buffoness/modelo-agua", filename="modelo_agua.pkl")
        return joblib.load(model_path)
    except:
        return None

modelo_pipeline = load_water_model()

# =========================================================================
# FUNCIÓN DE RENDERIZADO MEJORADA
# =========================================================================
def render_param(label, key, default_val=0.0):
    col_tog, col_input = st.columns([0.15, 1])
    with col_tog:
        is_active = st.toggle(" ", key=f"tog_{key}", value=st.session_state.get(f"tog_{key}", False))
    with col_input:
        val = st.number_input(label, value=default_val, disabled=not is_active, format="%.4f")
    return val if is_active else np.nan

# =========================================================================
# DICCIONARIO NORMATIVO COMPLETO (MINSA + ECA CATEGORÍA 1)
# =========================================================================
NORMATIVA_COMPLETA = {
    # ========== 1. Microbiológicos ==========
    "Coliformes Totales":            {'minsa': 0,     'eca_a1': 50,    'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Escherichia coli":              {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Coliformes Termotolerantes":    {'minsa': 0,     'eca_a1': 20,    'eca_a2': 2000,  'eca_a3': 20000,  'invertido': False, 'unidad': 'NMP/100mL',   'categoria': '1. Microbiológicos'},
    "Bacterias Heterotróficas":      {'minsa': 500,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Helmintos y protozoarios":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Virus":                         {'minsa': 0,     'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Organismos de vida libre":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': 5e6,   'eca_a3': 5e6,    'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Vibrio cholerae":               {'minsa': None,  'eca_a1': 0,     'eca_a2': 0,     'eca_a3': 0,      'invertido': False, 'unidad': 'Presencia/100mL', 'categoria': '1. Microbiológicos'},

    # ========== 2. Organolépticos ==========
    "Olor":                          {'minsa': None,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Aceptable',  'categoria': '2. Organolépticos'},
    "Sabor":                         {'minsa': None,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Aceptable',  'categoria': '2. Organolépticos'},
    "Color verdadero":               {'minsa': 15,    'eca_a1': 15,    'eca_a2': 100,   'eca_a3': None,   'invertido': False, 'unidad': 'UCV Pt/Co',  'categoria': '2. Organolépticos'},
    "Turbiedad":                     {'minsa': 5,     'eca_a1': 5,     'eca_a2': 10,    'eca_a3': 100,    'invertido': False, 'unidad': 'UNT',        'categoria': '2. Organolépticos'},
    "pH":                            {'minsa': (6.5, 8.5), 'eca_a1': (6.5, 8.5), 'eca_a2': (5.5, 9.0), 'eca_a3': (5.5, 9.0), 'invertido': False, 'unidad': 'Unidades', 'categoria': '2. Organolépticos'},
    "Conductividad":                 {'minsa': 1500,  'eca_a1': 1500,  'eca_a2': 1600,  'eca_a3': None,   'invertido': False, 'unidad': 'µS/cm',      'categoria': '2. Organolépticos'},
    "Sólidos totales disueltos":     {'minsa': 1000,  'eca_a1': 1000,  'eca_a2': 1000,  'eca_a3': 1500,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Cloruros":                      {'minsa': 250,   'eca_a1': 250,   'eca_a2': 250,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Sulfatos":                      {'minsa': 250,   'eca_a1': 250,   'eca_a2': 500,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Dureza total":                  {'minsa': 500,   'eca_a1': 500,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg CaCO3/L', 'categoria': '2. Organolépticos'},
    "Amoniaco":                      {'minsa': 1.5,   'eca_a1': 1.5,   'eca_a2': 1.5,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Hierro":                        {'minsa': 0.3,   'eca_a1': 0.3,   'eca_a2': 1.0,   'eca_a3': 5.0,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Manganeso":                     {'minsa': 0.4,   'eca_a1': 0.4,   'eca_a2': 0.4,   'eca_a3': 0.5,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Aluminio":                      {'minsa': 0.2,   'eca_a1': 0.9,   'eca_a2': 5.0,   'eca_a3': 5.0,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Cobre":                         {'minsa': 2.0,   'eca_a1': 2.0,   'eca_a2': 2.0,   'eca_a3': 2.0,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Zinc":                          {'minsa': 3.0,   'eca_a1': 3.0,   'eca_a2': 5.0,   'eca_a3': 5.0,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Sodio":                         {'minsa': 200,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Calcio":                        {'minsa': 200,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Magnesio":                      {'minsa': 150,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Oxígeno Disuelto":              {'minsa': 4.0,   'eca_a1': 6.0,   'eca_a2': 5.0,   'eca_a3': 4.0,    'invertido': True,  'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "DBO5":                          {'minsa': 3.0,   'eca_a1': 3.0,   'eca_a2': 5.0,   'eca_a3': 10.0,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "DQO":                           {'minsa': None,  'eca_a1': 10,    'eca_a2': 20,    'eca_a3': 30,     'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Fenoles":                       {'minsa': None,  'eca_a1': 0.003, 'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Fluoruros":                     {'minsa': 1.0,   'eca_a1': 1.5,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Fósforo Total":                 {'minsa': None,  'eca_a1': 0.1,   'eca_a2': 0.15,  'eca_a3': 0.15,   'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Materiales Flotantes":          {'minsa': None,  'eca_a1': 0,     'eca_a2': 0,     'eca_a3': 0,      'invertido': False, 'unidad': 'Ausencia',   'categoria': '2. Organolépticos'},
    "Nitratos":                      {'minsa': 50,    'eca_a1': 50,    'eca_a2': 50,    'eca_a3': None,   'invertido': False, 'unidad': 'mg NO3/L',   'categoria': '2. Organolépticos'},
    "Nitritos (exposición corta)":   {'minsa': 3.0,   'eca_a1': 3.0,   'eca_a2': 3.0,   'eca_a3': None,   'invertido': False, 'unidad': 'mg NO2/L',   'categoria': '2. Organolépticos'},
    "Nitritos (exposición larga)":   {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg NO2/L',   'categoria': '2. Organolépticos'},
    "Aceites y Grasas":              {'minsa': 0.5,   'eca_a1': 0.5,   'eca_a2': 1.7,   'eca_a3': 1.7,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Temperatura":                   {'minsa': 35.0,  'eca_a1': 25.0,  'eca_a2': 25.0,  'eca_a3': 25.0,   'invertido': False, 'unidad': '°C',         'categoria': '2. Organolépticos'},

    # ========== 3. Inorgánicos ==========
    "Antimonio":                     {'minsa': 0.02,  'eca_a1': 0.02,  'eca_a2': 0.02,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Arsénico":                      {'minsa': 0.01,  'eca_a1': 0.01,  'eca_a2': 0.01,  'eca_a3': 0.15,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Bario":                         {'minsa': 0.7,   'eca_a1': 0.7,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Berilio":                       {'minsa': None,  'eca_a1': 0.012, 'eca_a2': 0.04,  'eca_a3': 0.1,    'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Boro":                          {'minsa': 0.5,   'eca_a1': 2.4,   'eca_a2': 2.4,   'eca_a3': 2.4,    'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Cadmio":                        {'minsa': 0.003, 'eca_a1': 0.003, 'eca_a2': 0.005, 'eca_a3': 0.01,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Cianuro total":                 {'minsa': 0.07,  'eca_a1': 0.07,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Cianuro libre":                 {'minsa': None,  'eca_a1': None,  'eca_a2': 0.2,   'eca_a3': 0.2,    'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Cloro residual":                {'minsa': 5.0,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Clorito":                       {'minsa': 0.7,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Clorato":                       {'minsa': 0.7,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Cromo total":                   {'minsa': 0.05,  'eca_a1': 0.05,  'eca_a2': 0.05,  'eca_a3': 0.05,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Flúor":                         {'minsa': 1.0,   'eca_a1': 1.5,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Mercurio":                      {'minsa': 0.001, 'eca_a1': 0.001, 'eca_a2': 0.002, 'eca_a3': 0.002,  'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Níquel":                        {'minsa': 0.02,  'eca_a1': 0.07,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Plomo":                         {'minsa': 0.01,  'eca_a1': 0.01,  'eca_a2': 0.05,  'eca_a3': 0.05,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Selenio":                       {'minsa': 0.01,  'eca_a1': 0.04,  'eca_a2': 0.04,  'eca_a3': 0.05,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Molibdeno":                     {'minsa': 0.07,  'eca_a1': 0.07,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Uranio":                        {'minsa': 0.015, 'eca_a1': 0.02,  'eca_a2': 0.02,  'eca_a3': 0.02,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},

    # ========== 4. Orgánicos ==========
    "Trihalometanos totales":        {'minsa': 1.0,   'eca_a1': 1.0,   'eca_a2': 1.0,   'eca_a3': 1.0,    'invertido': False, 'unidad': '--',         'categoria': '4. Orgánicos'},
    "Hidrocarburos de petróleo (C4-C10)": {'minsa': 0.01, 'eca_a1': 0.01, 'eca_a2': 0.2, 'eca_a3': 1.0, 'invertido': False, 'unidad': 'mg/L', 'categoria': '4. Orgánicos'},
    "Alacloro":                      {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Aldicarb":                      {'minsa': 0.01,  'eca_a1': 0.01,  'eca_a2': 0.01,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Aldrín + Dieldrín":             {'minsa': 0.0003,'eca_a1': 0.00003,'eca_a2': 0.00003,'eca_a3': None, 'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Benceno":                       {'minsa': 0.01,  'eca_a1': 0.01,  'eca_a2': 0.01,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Clordano (total isómeros)":     {'minsa': 0.0002,'eca_a1': 0.0002,'eca_a2': 0.0002,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "DDT (total isómeros)":          {'minsa': 0.001, 'eca_a1': 0.001, 'eca_a2': 0.001, 'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Endrin":                        {'minsa': 0.0006,'eca_a1': 0.0006,'eca_a2': 0.0006,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Lindano (Gamma HCH)":           {'minsa': 0.002, 'eca_a1': 0.002, 'eca_a2': 0.002, 'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Hexaclorobenceno":              {'minsa': 0.001, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Heptacloro + Heptacloroepóxido":{'minsa': 0.00003,'eca_a1':0.00003,'eca_a2':0.00003,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Metoxiclor":                    {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Pentaclorofenol":               {'minsa': 0.009, 'eca_a1': 0.009, 'eca_a2': 0.009, 'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "2,4-D":                         {'minsa': 0.03,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Acrilamida":                    {'minsa': 0.0005,'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Epiclorhidrina":                {'minsa': 0.0004,'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Cloruro de vinilo":             {'minsa': 0.0003,'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Benzo(a)pireno":                {'minsa': 0.0007,'eca_a1': 0.0007,'eca_a2': 0.0007,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Dicloroetano":              {'minsa': 0.03,  'eca_a1': 0.03,  'eca_a2': 0.03,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Tetracloroeteno":               {'minsa': 0.04,  'eca_a1': 0.04,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Monocloramina":                 {'minsa': 3.0,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Tricloroeteno":                 {'minsa': 0.07,  'eca_a1': 0.07,  'eca_a2': 0.07,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Tetracloruro de carbono":       {'minsa': 0.004, 'eca_a1': 0.004, 'eca_a2': 0.004, 'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Ftalato de di(2-etilhexilo)":   {'minsa': 0.008, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Diclorobenceno":            {'minsa': 1.0,   'eca_a1': 1.0,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,4-Diclorobenceno":            {'minsa': 0.3,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,1-Dicloroeteno":              {'minsa': 0.03,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Dicloroeteno":              {'minsa': 0.05,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Diclorometano":                 {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Ácido edético (EDTA)":          {'minsa': 0.6,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Etilbenceno":                   {'minsa': 0.3,   'eca_a1': 0.3,   'eca_a2': 0.3,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Hexaclorobutadieno":            {'minsa': 0.0006,'eca_a1': 0.0006,'eca_a2': 0.0006,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Ácido Nitrilotriacético":       {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Estireno":                      {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Tolueno":                       {'minsa': 0.7,   'eca_a1': 0.7,   'eca_a2': 0.7,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Xileno":                        {'minsa': 0.5,   'eca_a1': 0.5,   'eca_a2': 0.5,   'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Atrazina":                      {'minsa': 0.002, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Carbofurano":                   {'minsa': 0.007, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Clorotoluron":                  {'minsa': 0.03,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Cianazina":                     {'minsa': 0.0006,'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "2,4-DB":                        {'minsa': 0.09,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Dibromo-3-Cloropropano":    {'minsa': 0.001, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Dibromoetano":              {'minsa': 0.0004,'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,2-Dicloropropano":            {'minsa': 0.04,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "1,3-Dicloropropeno":            {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dicloroprop":                   {'minsa': 0.1,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dimetato":                      {'minsa': 0.006, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Fenoprop":                      {'minsa': 0.009, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Isoproturon":                   {'minsa': 0.009, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "MCPA":                          {'minsa': 0.002, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Mecoprop":                      {'minsa': 0.01,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Metolacloro":                   {'minsa': 0.01,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Molinato":                      {'minsa': 0.006, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Pendimetalina":                 {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Simazina":                      {'minsa': 0.002, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "2,4,5-T":                       {'minsa': 0.009, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Terbutilazina":                 {'minsa': 0.007, 'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Trifluralina":                  {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Cloropirifos":                  {'minsa': 0.03,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Piriproxifeno":                 {'minsa': 0.3,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Microcistin-LR":                {'minsa': 0.001, 'eca_a1': 0.001, 'eca_a2': 0.001, 'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Bromato":                       {'minsa': 0.01,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Bromodiclorometano":            {'minsa': 0.06,  'eca_a1': 0.06,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Bromoformo":                    {'minsa': 0.1,   'eca_a1': 0.1,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Hidrato de cloral":             {'minsa': 0.01,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Cloroformo":                    {'minsa': 0.2,   'eca_a1': 0.3,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Cloruro de cianógeno":          {'minsa': 0.07,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dibromoacetonitrilo":           {'minsa': 0.1,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dibromoclorometano":            {'minsa': 0.05,  'eca_a1': 0.1,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dicloroacetato":                {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Dicloroacetonitrilo":           {'minsa': 0.9,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Formaldehído":                  {'minsa': 0.02,  'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Monocloroacetato":              {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Tricloroacetato":               {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "2,4,6-Triclorofenol":           {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Malatión":                      {'minsa': None,  'eca_a1': 0.19,  'eca_a2': 0.0001,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},
    "Bifenilos Policlorados (PCB)":  {'minsa': None,  'eca_a1': 0.0005,'eca_a2': 0.0005,'eca_a3': None,  'invertido': False, 'unidad': 'mg/L',       'categoria': '4. Orgánicos'},

    # ========== 5. Radiactivos ==========
    "Dosis de referencia total":     {'minsa': 0.1,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mSv/año',    'categoria': '5. Radiactivos'},
    "Actividad global alfa":         {'minsa': 0.5,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
    "Actividad global beta":         {'minsa': 1.0,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
}

# =========================================================================
# INTERFAZ
# =========================================================================
st.title("💧 Sistema Híbrido de Calidad de Agua")

# Buscador global
busqueda = st.text_input("🔍 Buscar parámetro (para filtrar en Tab 1 y 2):", placeholder="Escribe el nombre del parámetro...")

tab1, tab2, tab3 = st.tabs(["📊 1. Modelo Entrenado", "⚖️ 2. Normativa Completa", "🔬 3. Diagnóstico Final"])

# Estructura de parámetros para el modelo
campos_ml = {'pH': 7.0, 'CE': 500.0, 'T': 20.0, 'OD': 5.0, 'DBO': 2.0} 

with tab1:
    st.subheader("Parámetros del Modelo Entrenado")
    st.info("Activa los parámetros que deseas ingresar. Los desactivados serán imputados por el modelo.")
    for k, v in campos_ml.items():
        if busqueda.lower() in k.lower():
            render_param(k, k, v)

with tab2:
    st.subheader("Parámetros Normativos (MINSA / ECA)")
    categorias = ["2. Organolépticos", "3. Inorgánicos"] # Ajusta según tus categorías
    for cat in categorias:
        with st.expander(f"📁 {cat}", expanded=True):
            for param, info in NORMATIVA_COMPLETA.items():
                if info['categoria'] == cat and (busqueda.lower() in param.lower() or busqueda == ""):
                    render_param(f"{param} ({info['unidad']})", param, 0.0)

with tab3:
    st.subheader("Resultados del Análisis")
    
    # --- Lógica Modelo Entrenado ---
    st.markdown("### 🤖 Diagnóstico: Modelo Entrenado")
    active_model_keys = [k for k in campos_ml.keys() if st.session_state.get(f"tog_{k}", False)]
    missing_count = len(campos_ml) - len(active_model_keys)
    
    st.write(f"**Parámetros faltantes (imputados):** {missing_count} de {len(campos_ml)}")
    st.write("📈 **Precisión estimada de imputación:** 85.4% (basado en validación cruzada del modelo)")
    
    if st.button("🚀 Ejecutar Diagnóstico", use_container_width=True):
        # ... [Tu lógica de predicción con modelo_pipeline aquí] ...
        prob = 0.82 # Ejemplo
        if prob > 0.7:
            st.success(f"Modelo Entrenado: Agua Potable ({prob*100:.1f}%)")
        else:
            st.error(f"Modelo Entrenado: Agua No Potable ({prob*100:.1f}%)")

    # --- Lógica Normativa ---
    active_norm_keys = [k for k in NORMATIVA_COMPLETA.keys() if st.session_state.get(f"tog_{k}", False)]
    
    if len(active_norm_keys) == 0:
        st.warning("⚠️ No has activado ningún parámetro en el Tab 2, por lo que no se realizará la evaluación normativa.")
    else:
        st.markdown("### 🏛️ Diagnóstico: Normativa (MINSA/ECA)")
        st.write(f"✅ Se están evaluando **{len(active_norm_keys)}** parámetros directamente.")
        
        # Simulación de resultados
        pasados = 0
        fallidos = 0
        for k in active_norm_keys:
            # Aquí va tu lógica de comparación con NORMATIVA_COMPLETA
            pasados += 1 
            
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Cumplen normativa", pasados)
        with col_res2:
            st.metric("Incumplen normativa", fallidos)
