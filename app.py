import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# =========================================================================
# CONFIGURACIÓN VISUAL DE LA APP
# =========================================================================
st.set_page_config(page_title="Sistema Híbrido - Calidad del Agua", page_icon="💧", layout="wide")

st.markdown("""
<style>
    .card-potable { padding: 20px; background-color: #D1E7DD; border-radius: 10px; border-left: 8px solid #0F5132; color: #0F5132; }
    .card-nopotable { padding: 20px; background-color: #F8D7DA; border-radius: 10px; border-left: 8px solid #842029; color: #842029; }
    .treatment-box { padding: 15px; background-color: #FFF3CD; border-radius: 8px; border-left: 5px solid #FFC107; color: #664D03; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("💧 Sistema Híbrido de Diagnóstico de Calidad del Agua")
st.markdown("### Validación Normativa Multi-nivel (MINSA/ECA) + Inteligencia Artificial")

# =========================================================================
# 2. FUNCIÓN DE CARGA DEL MODELO DESDE HUGGING FACE
# =========================================================================
@st.cache_resource
def load_water_model():
    try:
        model_path = hf_hub_download(repo_id="buffoness/modelo-agua", filename="modelo_agua.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo desde Hugging Face: {e}")
        return None

modelo_pipeline = load_water_model()

# =========================================================================
# 3. DICCIONARIOS NORMATIVOS COMPLETOS (MINSA + ECA CATEGORÍA 1)
# =========================================================================
# Estructura: {nombre_param: {'minsa': LMP, 'eca_a1': LMP_A1, 'eca_a2': LMP_A2, 'eca_a3': LMP_A3, 'invertido': bool, 'unidad': str, 'categoria': str}}
# Si un límite no aplica se usa None. Para pH se usa tupla (min, max).

NORMATIVA_COMPLETA = {
    # ----------------------------------- MICROBIOLÓGICOS -----------------------------------
    "Coliformes Totales":            {'minsa': 0,     'eca_a1': 50,    'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Escherichia coli":              {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Coliformes Termotolerantes":    {'minsa': 0,     'eca_a1': 20,    'eca_a2': 2000,  'eca_a3': 20000,  'invertido': False, 'unidad': 'NMP/100mL',   'categoria': '1. Microbiológicos'},
    "Bacterias Heterotróficas":      {'minsa': 500,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Helmintos y protozoarios":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Virus":                         {'minsa': 0,     'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Organismos de vida libre":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': 5e6,   'eca_a3': 5e6,    'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Vibrio cholerae":               {'minsa': None,  'eca_a1': 0,     'eca_a2': 0,     'eca_a3': 0,      'invertido': False, 'unidad': 'Presencia/100mL', 'categoria': '1. Microbiológicos'},

    # ----------------------------------- ORGANOLÉPTICOS -----------------------------------
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

    # ----------------------------------- INORGÁNICOS -----------------------------------
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

    # ----------------------------------- ORGÁNICOS (se incluyen todos los listados) -----------------------------------
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

    # ----------------------------------- RADIACTIVOS -----------------------------------
    "Dosis de referencia total":     {'minsa': 0.1,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mSv/año',    'categoria': '5. Radiactivos'},
    "Actividad global alfa":         {'minsa': 0.5,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
    "Actividad global beta":         {'minsa': 1.0,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
}

# Definir los 14 parámetros que alimentan al modelo de IA
FEATURES_ML = ['pH', 'CE', 'T', 'OD', 'DBO', 'CT', 'AyG', 'ArT', 'PbT', 'CuT', 'MnT', 'Ca', 'Mg', 'Dureza']

# Mapeo de nombres ML a nombres en la normativa (si difieren)
MAP_ML_NORMATIVA = {
    'pH': 'pH',
    'CE': 'Conductividad',
    'T': 'Temperatura',
    'OD': 'Oxígeno Disuelto',
    'DBO': 'DBO5',
    'CT': 'Coliformes Termotolerantes',
    'AyG': 'Aceites y Grasas',
    'ArT': 'Arsénico',
    'PbT': 'Plomo',
    'CuT': 'Cobre',
    'MnT': 'Manganeso',
    'Ca': 'Calcio',
    'Mg': 'Magnesio',
    'Dureza': 'Dureza total'
}

# =========================================================================
# 4. INTERFAZ DE USUARIO EN PESTAÑAS
# =========================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 14 Parámetros Base (ML)", 
    "⚖️ Normativa Completa (MINSA/ECA)", 
    "🔬 Diagnóstico y Tratamiento"
])

# Diccionario global para todos los valores ingresados
data_input = {}

with tab1:
    st.subheader("Parámetros Fisicoquímicos Principales (modelo IA)")
    col1, col2, col3 = st.columns(3)
    
    defaults_ml = {
        'pH': 7.5, 'CE': 400.0, 'T': 18.0, 'OD': 7.5, 'DBO': 2.0,
        'CT': 10.0, 'AyG': 0.1, 'ArT': 0.002, 'PbT': 0.001, 'CuT': 0.005,
        'MnT': 0.02, 'Ca': 45.0, 'Mg': 12.0, 'Dureza': 150.0
    }
    
    with col1:
        data_input['pH'] = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5, step=0.1, key="ml_ph")
        data_input['CE'] = st.number_input("Conductividad (µS/cm)", min_value=0.0, value=400.0, step=10.0, key="ml_ce")
        data_input['T'] = st.number_input("Temperatura (°C)", min_value=0.0, value=18.0, step=0.5, key="ml_t")
        data_input['OD'] = st.number_input("Oxígeno Disuelto (mg/L)", min_value=0.0, value=7.5, step=0.1, key="ml_od")
        data_input['DBO'] = st.number_input("DBO5 (mg/L)", min_value=0.0, value=2.0, step=0.1, key="ml_dbo")
    with col2:
        data_input['CT'] = st.number_input("Coliformes Termotolerantes (NMP/100mL)", min_value=0.0, value=10.0, step=1.0, key="ml_ct")
        data_input['AyG'] = st.number_input("Aceites y Grasas (mg/L)", min_value=0.0, value=0.1, step=0.1, key="ml_ayg")
        data_input['ArT'] = st.number_input("Arsénico Total (mg/L)", min_value=0.0, format="%.5f", value=0.002, key="ml_art")
        data_input['PbT'] = st.number_input("Plomo Total (mg/L)", min_value=0.0, format="%.5f", value=0.001, key="ml_pbt")
        data_input['CuT'] = st.number_input("Cobre Total (mg/L)", min_value=0.0, format="%.5f", value=0.005, key="ml_cut")
    with col3:
        data_input['MnT'] = st.number_input("Manganeso Total (mg/L)", min_value=0.0, format="%.5f", value=0.02, key="ml_mnt")
        data_input['Ca'] = st.number_input("Calcio (mg/L)", min_value=0.0, value=45.0, step=1.0, key="ml_ca")
        data_input['Mg'] = st.number_input("Magnesio (mg/L)", min_value=0.0, value=12.0, step=1.0, key="ml_mg")
        data_input['Dureza'] = st.number_input("Dureza Total (mg CaCO3/L)", min_value=0.0, value=150.0, key="ml_dureza")

with tab2:
    st.subheader("Parámetros Legales Adicionales (MINSA / ECA Categoría 1)")
    st.markdown("Solo se muestran los parámetros normativos **no incluidos** en la pestaña anterior.")
    
    # Agrupar por categoría y excluir los que ya están en los 14 parámetros ML
    normativas_extra = {}
    nombres_ml_norm = set(MAP_ML_NORMATIVA.values())  # nombres que ya están en la pestaña 1
    
    for param, info in NORMATIVA_COMPLETA.items():
        if param not in nombres_ml_norm:  # no duplicar
            cat = info['categoria']
            normativas_extra.setdefault(cat, {})[param] = info
    
    # Crear un expander por cada categoría
    for categoria, parametros in normativas_extra.items():
        with st.expander(f"📁 {categoria}"):
            cols = st.columns(4)
            for i, (param, info) in enumerate(parametros.items()):
                with cols[i % 4]:
                    # Usar un valor por defecto apropiado (0.0 excepto para algunos)
                    default_val = 0.0
                    if param == "Cloro residual":
                        default_val = 1.0
                    elif param == "Oxígeno Disuelto":  # ya no debería aparecer, pero por si acaso
                        default_val = 7.0
                    data_input[param] = st.number_input(
                        f"{param} ({info['unidad']})",
                        value=default_val,
                        format="%.4f",
                        key=f"extra_{param}"
                    )

with tab3:
    st.subheader("🔍 Resultados e Informe Técnico")
    
    if st.button("🚀 Procesar Diagnóstico Completo", use_container_width=True):
        # --- EVALUACIÓN NORMATIVA MINSA ---
        infracciones_minsa = []
        for param, info in NORMATIVA_COMPLETA.items():
            if param not in data_input or data_input[param] is None:
                continue
            minsa_lim = info['minsa']
            if minsa_lim is None:
                continue
            valor = data_input[param]
            # Verificar tipo de límite
            if param == 'pH' or (isinstance(minsa_lim, tuple) and len(minsa_lim) == 2):
                low, high = minsa_lim
                if not (low <= valor <= high):
                    infracciones_minsa.append((param, valor, f"Rango {low} - {high}"))
            elif info.get('invertido', False):
                # Límite inferior (debe ser >= límite)
                if valor < minsa_lim:
                    infracciones_minsa.append((param, valor, f"Mínimo permitido {minsa_lim}"))
            else:
                if valor > minsa_lim:
                    infracciones_minsa.append((param, valor, f"Máximo permitido {minsa_lim}"))
        
        es_potable_minsa = len(infracciones_minsa) == 0
        
        # --- EVALUACIÓN ECA (categorías A1, A2, A3) ---
        peor_categoria_eca = "A1"
        detalles_eca = []
        
        if not es_potable_minsa:
            for param, info in NORMATIVA_COMPLETA.items():
                if param not in data_input or data_input[param] is None:
                    continue
                valor = data_input[param]
                # Evaluar solo si al menos un límite ECA existe
                eca_vals = []
                for subcat in ['eca_a1', 'eca_a2', 'eca_a3']:
                    lim = info.get(subcat)
                    if lim is not None:
                        eca_vals.append((subcat, lim))
                if not eca_vals:
                    continue
                
                # Determinar la categoría para este parámetro
                cat = None
                invertido = info.get('invertido', False)
                # Recorrer en orden A1 -> A2 -> A3
                for subcat, lim in [('eca_a1', info['eca_a1']), ('eca_a2', info['eca_a2']), ('eca_a3', info['eca_a3'])]:
                    if lim is None:
                        continue
                    if param == 'pH' or isinstance(lim, tuple):
                        low, high = lim
                        if low <= valor <= high:
                            cat = subcat.replace('eca_', '').upper()
                            break
                    elif invertido:
                        if valor >= lim:
                            cat = subcat.replace('eca_', '').upper()
                            break
                    else:
                        if valor <= lim:
                            cat = subcat.replace('eca_', '').upper()
                            break
                if cat is None:
                    cat = "EXCEDE A3"
                
                detalles_eca.append({'Parámetro': param, 'Valor': valor, 'Categoría ECA': cat})
                # Actualizar la peor categoría global
                orden = {"A1": 1, "A2": 2, "A3": 3, "EXCEDE A3": 4}
                if orden[cat] > orden[peor_categoria_eca]:
                    peor_categoria_eca = cat
        
        # --- PREDICCIÓN MACHINE LEARNING ---
        ml_result = "Desactivado (modelo no disponible)"
        prob_potable = 0.0
        if modelo_pipeline is not None:
            try:
                # Construir vector de entrada con los 14 parámetros en el orden correcto
                features_vector = pd.DataFrame([[data_input[feat] for feat in FEATURES_ML]], columns=FEATURES_ML)
                prob = modelo_pipeline.predict_proba(features_vector)[0]
                prob_potable = prob[1]  # probabilidad de ser potable
                pred = 1 if prob_potable >= 0.75 else 0
                if pred == 1:
                    ml_result = f"Potable (Prob: {prob_potable*100:.1f}%)"
                else:
                    ml_result = f"No Potable (Prob: {prob[0]*100:.1f}%)"
            except Exception as e:
                ml_result = f"Error en IA: {e}"
        
        # --- VISUALIZACIÓN DE RESULTADOS ---
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### 🏛️ Evaluación Legal (MINSA)")
            if es_potable_minsa:
                st.markdown("""<div class="card-potable"><h4>✅ CUMPLE MINSA</h4><p>Apto para consumo directo.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card-nopotable"><h4>❌ NO CUMPLE MINSA</h4><p>Violación en {len(infracciones_minsa)} parámetros.</p></div>""", unsafe_allow_html=True)
                with st.expander("Ver parámetros que incumplen MINSA"):
                    st.dataframe(pd.DataFrame(infracciones_minsa, columns=['Variable', 'Valor', 'Límite']))
        
        with col_res2:
            st.markdown("#### 🤖 Predicción IA (Random Forest / GB)")
            if modelo_pipeline is not None:
                st.metric(label="Clasificador (Umbral Seguridad 0.75)", value=ml_result)
                st.progress(float(prob_potable), text="Probabilidad general de Potabilidad")
        
        # --- RECOMENDACIÓN DE TRATAMIENTO ECA ---
        if not es_potable_minsa:
            st.markdown("---")
            st.markdown(f"### 🛠️ Plan de Tratamiento según ECA (Peor categoría detectada: {peor_categoria_eca})")
            
            if peor_categoria_eca == "A1":
                st.markdown("""<div class="treatment-box">🟢 <b>Subcategoría A1: Desinfección Simple.</b><br>Requiere cloración estándar u ozonización para eliminar carga microbiológica antes de distribuir.</div>""", unsafe_allow_html=True)
            elif peor_categoria_eca == "A2":
                st.markdown("""<div class="treatment-box" style="border-left-color: #FD7E14; background-color: #FFE8D6;">🟠 <b>Subcategoría A2: Tratamiento Convencional.</b><br>Implementar Coagulación, Floculación, Sedimentación, Filtración rápida y Desinfección.</div>""", unsafe_allow_html=True)
            elif peor_categoria_eca == "A3":
                st.markdown("""<div class="treatment-box" style="border-left-color: #DC3545; background-color: #F8D7DA;">🔴 <b>Subcategoría A3: Tratamiento Avanzado.</b><br>Contaminación alta. Requiere Ósmosis Inversa, Carbón Activado o Procesos de Oxidación Avanzada.</div>""", unsafe_allow_html=True)
            else:  # EXCEDE A3
                st.markdown("""<div class="treatment-box" style="background-color: #343A40; color: #FFF; border-left-color: #000;">⚫ <b>EXCEDE A3: ALERTA CRÍTICA.</b><br>El agua supera la capacidad de las plantas potabilizadoras convencionales. Descarte inmediato para consumo.</div>""", unsafe_allow_html=True)
            
            with st.expander("Ver desglose completo de categorías ECA"):
                st.dataframe(pd.DataFrame(detalles_eca))
