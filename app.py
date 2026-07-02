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
    /* Toggle y campos */
    .stToggle { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .card-res { padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    
    /* Tarjetas de resultado */
    .card-potable { padding: 20px; background-color: #D1E7DD; border-radius: 10px; border-left: 8px solid #0F5132; color: #0F5132; }
    .card-nopotable { padding: 20px; background-color: #F8D7DA; border-radius: 10px; border-left: 8px solid #842029; color: #842029; }
    .treatment-box { padding: 15px; background-color: #FFF3CD; border-radius: 8px; border-left: 5px solid #FFC107; color: #664D03; font-weight: bold;}

    /* Alerta de peligro (excede A3) - estilizada como los errores de Streamlit */
    .danger-box {
        padding: 1rem;
        background-color: #fdecea;
        border-left: 4px solid #ff6b6b;
        color: #3e2723;
        font-weight: bold;
        border-radius: 0.5rem;
        margin: 15px 0;
    }

    /* Centrar tablas y contenido */
    [data-testid="stTable"] table,
    [data-testid="stDataFrame"] table {
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
    }
    th, td {
        text-align: center !important;
    }

    /* Títulos centrados */
    h3, h4 {
        text-align: center;
    }
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
# DICCIONARIO NORMATIVO COMPLETO (MINSA + ECA Categoría 1)
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
    "Cloruros":                      {'minsa': 250,   'eca_a1': 250,   'eca_a2': 250,   'eca_a3': 250,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
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
    "Nitratos":                      {'minsa': 50,    'eca_a1': 50,    'eca_a2': 50,    'eca_a3': 50,     'invertido': False, 'unidad': 'mg NO3/L',   'categoria': '2. Organolépticos'},
    "Nitritos (exposición corta)":   {'minsa': 3.0,   'eca_a1': 3.0,   'eca_a2': 3.0,   'eca_a3': None,   'invertido': False, 'unidad': 'mg NO2/L',   'categoria': '2. Organolépticos'},
    "Nitritos (exposición larga)":   {'minsa': 0.2,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg NO2/L',   'categoria': '2. Organolépticos'},
    "Aceites y Grasas":              {'minsa': 0.5,   'eca_a1': 0.5,   'eca_a2': 1.7,   'eca_a3': 1.7,    'invertido': False, 'unidad': 'mg/L',       'categoria': '2. Organolépticos'},
    "Temperatura":                   {'minsa': 35.0,  'eca_a1': 25.0,  'eca_a2': 25.0,  'eca_a3': 25.0,   'invertido': False, 'unidad': '°C',         'categoria': '2. Organolépticos'},

    # ========== 3. Inorgánicos ==========
    "Antimonio":                     {'minsa': 0.02,  'eca_a1': 0.02,  'eca_a2': 0.02,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Arsénico":                      {'minsa': 0.01,  'eca_a1': 0.01,  'eca_a2': 0.01,  'eca_a3': 0.15,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Bario":                         {'minsa': 0.7,   'eca_a1': 0.7,   'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Berilio":                       {'minsa': None,  'eca_a1': 0.012, 'eca_a2': 0.04,  'eca_a3': 0.1,    'invertido': False, 'unidad': 'mg/L',       'categoria': '3. Inorgánicos'},
    "Boro":                          {'minsa': 1.5,   'eca_a1': 2.4,   'eca_a2': 2.4,   'eca_a3': 2.4,    'invertido': False, 'unidad': 'mg B/L',     'categoria': '3. Inorgánicos'},
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

    # ========== 4. Orgánicos (lista completa) ==========
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
# GESTIÓN DE ESTADO (SESSION STATE)
# =========================================================================
campos_ml = {
    'pH': 7.0, 'CE': 0.0, 'T': 20.0, 'OD': 5.0, 'DBO': 0.0,
    'CT': 0.0, 'AyG': 0.0, 'ArT': 0.0, 'PbT': 0.0, 'CuT': 0.0,
    'MnT': 0.0, 'Ca': 0.0, 'Mg': 0.0, 'Dureza': 0.0
}

nombres_modelo_ui = {
    'pH': 'pH (Potencial de Hidrógeno)',
    'CE': 'Conductividad Eléctrica (µS/cm)',
    'T': 'Temperatura (°C)',
    'OD': 'Oxígeno Disuelto (mg/L)',
    'DBO': 'Demanda Bioquímica de Oxígeno - DBO5 (mg/L)',
    'CT': 'Coliformes Totales (NMP/100 mL)',
    'AyG': 'Aceites y Grasas (mg/L)',
    'ArT': 'Arsénico Total (mg/L)',
    'PbT': 'Plomo Total (mg/L)',
    'CuT': 'Cobre Total (mg/L)',
    'MnT': 'Manganeso Total (mg/L)',
    'Ca': 'Calcio (mg/L)',
    'Mg': 'Magnesio (mg/L)',
    'Dureza': 'Dureza total (mg CaCO3/L)'
}

def inicializar_estado():
    for k in campos_ml:
        if f"tog_{k}" not in st.session_state:
            st.session_state[f"tog_{k}"] = False
        if f"val_{k}" not in st.session_state:
            st.session_state[f"val_{k}"] = campos_ml[k]

    for k in NORMATIVA_COMPLETA.keys():
        if f"tog_{k}" not in st.session_state:
            st.session_state[f"tog_{k}"] = False
        if f"val_{k}" not in st.session_state:
            st.session_state[f"val_{k}"] = 0.0

inicializar_estado()

# =========================================================================
# FUNCIÓN AUXILIAR PARA RENDERIZAR PARÁMETROS
# =========================================================================
def render_param(label, key, default_val=0.0):
    col1, col2 = st.columns([0.1, 1])
    with col1:
        st.toggle(" ", key=f"tog_{key}")
    with col2:
        is_active = st.session_state[f"tog_{key}"]
        st.number_input(label, value=st.session_state.get(f"val_{key}", default_val), 
                        disabled=not is_active, format="%.4f", key=f"val_{key}")

# =========================================================================
# INTERFAZ
# =========================================================================
st.title("💧 Diagnóstico Híbrido")

tab1, tab2, tab3 = st.tabs(["📊 1. Inputs IA", "⚖️ 2. Normativa Completa", "🔬 3. Diagnóstico Final"])

with tab1:
    st.subheader("Parámetros del Modelo (IA)")
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("✅ Activar todos", key="btn_act_t1"):
        for k in campos_ml: st.session_state[f"tog_{k}"] = True
        st.rerun()
    if col_btn2.button("❌ Desactivar todos", key="btn_des_t1"):
        for k in campos_ml: st.session_state[f"tog_{k}"] = False
        st.rerun()
        
    st.write("---")
    
    for k, v in campos_ml.items():
        render_param(nombres_modelo_ui[k], k, v)

with tab2:
    st.subheader("Parámetros Normativos (MINSA / ECA)")
    categorias = ["1. Microbiológicos", "2. Organolépticos", "3. Inorgánicos", "4. Orgánicos", "5. Radiactivos"]
    ya_incluidos = {'pH', 'Conductividad', 'Temperatura', 'Oxígeno Disuelto', 'DBO5',
                    'Coliformes Termotolerantes', 'Aceites y Grasas', 'Arsénico',
                    'Plomo', 'Cobre', 'Manganeso', 'Calcio', 'Magnesio', 'Dureza total'}
    
    for cat in categorias:
        with st.expander(cat, expanded=False):
            params_cat = [p for p, info in NORMATIVA_COMPLETA.items() if info['categoria'] == cat and p not in ya_incluidos]
            
            c1, c2 = st.columns(2)
            if c1.button(f"✅ Activar categoría", key=f"act_{cat}"):
                for p in params_cat: st.session_state[f"tog_{p}"] = True
                st.rerun()
            if c2.button(f"❌ Desactivar categoría", key=f"des_{cat}"):
                for p in params_cat: st.session_state[f"tog_{p}"] = False
                st.rerun()
            
            for param in params_cat:
                info = NORMATIVA_COMPLETA[param]
                render_param(f"{param} ({info['unidad']})", param, 0.0)

with tab3:
    st.subheader("Resultados")
    
    active_model_keys = [k for k in campos_ml.keys() if st.session_state.get(f"tog_{k}", False)]
    active_norm_keys = [k for k in NORMATIVA_COMPLETA.keys() if st.session_state.get(f"tog_{k}", False) and k not in ya_incluidos]
    
    model_problematicos = []
    for param in active_model_keys:
        valor = st.session_state.get(f"val_{param}", 0.0)
        nombre_norma = None
        for n_k in NORMATIVA_COMPLETA.keys():
            if param.lower() in n_k.lower() or n_k.lower() in param.lower():
                nombre_norma = n_k
                break
        
        if nombre_norma:
            limite = NORMATIVA_COMPLETA[nombre_norma]['minsa']
            if limite is not None:
                if isinstance(limite, tuple):
                    if not (limite[0] <= valor <= limite[1]): model_problematicos.append(param)
                elif NORMATIVA_COMPLETA[nombre_norma].get('invertido', False):
                    if valor < limite: model_problematicos.append(param)
                elif valor > limite:
                    model_problematicos.append(param)

    if st.button("🚀 Ejecutar Diagnóstico", use_container_width=True):
        # 1. PREDICCIÓN IA
        st.markdown("### 🤖 Predicción de IA")
        st.write(f"**Parámetros seleccionados:** {len(active_model_keys)} de {len(campos_ml)}")
        
        if modelo_pipeline:
            inputs_globales = {k: (st.session_state.get(f"val_{k}", 0.0) if st.session_state.get(f"tog_{k}", False) else np.nan) for k in campos_ml.keys()}
            features_model = ['pH', 'CE', 'T', 'OD', 'DBO', 'CT', 'AyG', 'ArT', 'PbT', 'CuT', 'MnT', 'Ca', 'Mg', 'Dureza']
            df_ia = pd.DataFrame([inputs_globales])[features_model]
            
            try:
                prob = modelo_pipeline.predict_proba(df_ia)[0][1]
                st.metric("Confianza / Probabilidad de Potabilidad", f"{prob*100:.2f}%")
                
                if model_problematicos:
                    nombres_problematicos = [nombres_modelo_ui.get(p, p) for p in model_problematicos]
                    st.warning(f"Parámetros que afectan negativamente la potabilidad detectados: {', '.join(nombres_problematicos)}")
                
                if prob > 0.75:
                    st.success("✅ IA: Apto para consumo (umbral > 0.75)")
                else:
                    st.error("❌ IA: No apto según el modelo predictivo")
            except Exception as e:
                st.warning(f"No se pudo ejecutar la IA adecuadamente (El modelo imputará automáticamente). Error detallado: {e}")
        else:
            st.info("Modelo no cargado. Verifica la conexión a Hugging Face.")

        st.markdown("---")

        # 2. EVALUACIÓN NORMATIVA MINSA / ECA
        st.markdown("### 🏛️ Evaluación Normativa")
        if not active_norm_keys:
            st.info("ℹ️ En el apartado de normativas (Tab 2) no se activó ningún parámetro para su evaluación.")
        else:
            st.write(f"Evaluando **{len(active_norm_keys)}** parámetros extra normativos:")
            
            # --- EVALUACIÓN MINSA ---
            incumplimientos_minsa = []
            for param in active_norm_keys:
                valor = st.session_state.get(f"val_{param}", 0.0)
                info = NORMATIVA_COMPLETA[param]
                minsa_lim = info['minsa']
                if minsa_lim is None: continue
                
                if isinstance(minsa_lim, tuple):
                    low, high = minsa_lim
                    if not (low <= valor <= high):
                        incumplimientos_minsa.append((param, valor, f"{low} - {high}"))
                elif info.get('invertido', False):
                    if valor < minsa_lim:
                        incumplimientos_minsa.append((param, valor, f">= {minsa_lim}"))
                else:
                    if minsa_lim == 0:
                        if valor > 0:
                            incumplimientos_minsa.append((param, valor, f"<= {minsa_lim}"))
                    else:
                        if valor > minsa_lim:
                            incumplimientos_minsa.append((param, valor, f"<= {minsa_lim}"))
            
            if incumplimientos_minsa:
                st.error(f"⚠️ {len(incumplimientos_minsa)} parámetros sobrepasan los límites MINSA:")
                df_inc = pd.DataFrame(incumplimientos_minsa, columns=['Parámetro', 'Valor Ingresado', 'Límite Normativo'])
                st.dataframe(df_inc)
            else:
                st.success("✅ Todos los parámetros extra evaluados CUMPLEN con MINSA.")

            # --- EVALUACIÓN ECA ---
            st.markdown("#### 🌿 Clasificación ECA (Categoría 1)")
            peor_cat = "A1"
            detalles_eca = []
            orden_eca = {"A1": 1, "A2": 2, "A3": 3, "EXCEDE A3": 4}
            
            for param in active_norm_keys:
                valor = st.session_state.get(f"val_{param}", 0.0)
                info = NORMATIVA_COMPLETA[param]
                cat = None
                invertido = info.get('invertido', False)
                
                for subcat in ['eca_a1', 'eca_a2', 'eca_a3']:
                    lim = info.get(subcat)
                    if lim is None: continue
                    
                    if isinstance(lim, tuple):
                        if lim[0] <= valor <= lim[1]:
                            cat = subcat.replace('eca_', '').upper()
                            break
                    elif invertido:
                        if valor >= lim:
                            cat = subcat.replace('eca_', '').upper()
                            break
                    else:
                        if lim == 0:
                            if valor <= 0:
                                cat = subcat.replace('eca_', '').upper()
                                break
                        else:
                            if valor <= lim:
                                cat = subcat.replace('eca_', '').upper()
                                break
                if cat is None:
                    cat = "EXCEDE A3"

                limite_a3 = info.get('eca_a3', 'N/A')
                detalles_eca.append({
                    'Parámetro': param,
                    'Valor Ingresado': valor,
                    'Categoría Asignada': cat,
                    'Límite ECA (A3)': limite_a3 if limite_a3 is not None else 'N/A'
                })
                if orden_eca.get(cat, 0) > orden_eca.get(peor_cat, 0):
                    peor_cat = cat

            st.write(f"**Peor nivel detectado en ECA:** `{peor_cat}`")
            
            df_eca = pd.DataFrame(detalles_eca)
            df_eca = df_eca[['Parámetro', 'Valor Ingresado', 'Límite ECA (A3)', 'Categoría Asignada']]
            st.dataframe(df_eca)

            excedentes = [d for d in detalles_eca if d['Categoría Asignada'] == 'EXCEDE A3']
            if excedentes:
                st.markdown("""
                <div class="danger-box">
                    ⚠️ <b>ALERTA CRÍTICA:</b> Existen parámetros que superan el límite de la categoría A3 (tratamiento avanzado).<br>
                    Esta agua <b>no es apta</b> para potabilización convencional ni avanzada. Se requiere una fuente alternativa o procesos especializados.
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ Parámetros que EXCEDEN la categoría A3 (tratamiento avanzado):")
                df_exc = pd.DataFrame(excedentes)
                df_exc = df_exc[['Parámetro', 'Valor Ingresado', 'Límite ECA (A3)', 'Categoría Asignada']]
                st.table(df_exc)
