import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Evaluación de Calidad de Agua", layout="wide")

# ==========================================
# CARGA DEL MODELO IA
# ==========================================
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('modelo_agua.pkl', 'rb'))
    except Exception as e:
        return None

modelo = load_model()

# ==========================================
# LISTA COMPLETA DE PARÁMETROS (CON UNIDADES)
# ==========================================
# Se incluyen todos los parámetros normativos del MINSA y ECA.
# Los 14 primeros son los que utiliza el modelo de IA (orden exacto).
PARAMETROS_NOMBRES = [
    # --- Parámetros del modelo (14) ---
    "pH (Potencial de Hidrógeno)",
    "Turbidez (Unidades Nefelométricas - NTU)",
    "Conductividad Eléctrica (µS/cm)",
    "Sólidos Totales Disueltos - TDS (mg/L)",
    "Temperatura (°C)",
    "Oxígeno Disuelto (mg/L)",
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)",
    "Nitratos (mg/L)",
    "Coliformes Totales (NMP/100 mL)",
    "Coliformes Termotolerantes (NMP/100 mL)",
    "Plomo Total (mg/L)",
    "Arsénico Total (mg/L)",
    "Hierro Total (mg/L)",
    "Sulfatos (mg/L)",

    # --- Microbiológicos adicionales ---
    "Escherichia coli (UFC/100mL)",
    "Bacterias Heterotróficas (UFC/mL)",
    "Helmintos y protozoarios (N° org/L)",
    "Virus (UFC/mL)",
    "Organismos de vida libre (N° org/L)",
    "Vibrio cholerae (Presencia/100mL)",

    # --- Organolépticos adicionales ---
    "Olor (Aceptable)",
    "Sabor (Aceptable)",
    "Color verdadero (UCV Pt/Co)",
    "Cloruros (mg/L)",
    "Dureza total (mg CaCO3/L)",
    "Amoniaco (mg/L)",
    "Manganeso Total (mg/L)",
    "Aluminio (mg/L)",
    "Cobre Total (mg/L)",
    "Zinc (mg/L)",
    "Sodio (mg/L)",
    "Calcio (mg/L)",
    "Magnesio (mg/L)",
    "DQO (mg/L)",
    "Fenoles (mg/L)",
    "Fluoruros (mg/L)",
    "Fósforo Total (mg/L)",
    "Materiales Flotantes (Ausencia)",
    "Nitritos (exposición corta) (mg NO2/L)",
    "Nitritos (exposición larga) (mg NO2/L)",
    "Aceites y Grasas (mg/L)",

    # --- Inorgánicos ---
    "Antimonio (mg/L)",
    "Bario (mg/L)",
    "Berilio (mg/L)",
    "Boro (mg/L)",
    "Cadmio Total (mg/L)",
    "Cianuro total (mg/L)",
    "Cianuro libre (mg/L)",
    "Cloro residual (mg/L)",
    "Clorito (mg/L)",
    "Clorato (mg/L)",
    "Cromo total (mg/L)",
    "Flúor (mg/L)",
    "Mercurio Total (mg/L)",
    "Níquel (mg/L)",
    "Selenio (mg/L)",
    "Molibdeno (mg/L)",
    "Uranio (mg/L)",

    # --- Orgánicos (lista completa) ---
    "Trihalometanos totales (--)" ,
    "Hidrocarburos de petróleo (C4-C10) (mg/L)",
    "Alacloro (mg/L)",
    "Aldicarb (mg/L)",
    "Aldrín + Dieldrín (mg/L)",
    "Benceno (mg/L)",
    "Clordano (total isómeros) (mg/L)",
    "DDT (total isómeros) (mg/L)",
    "Endrin (mg/L)",
    "Lindano (Gamma HCH) (mg/L)",
    "Hexaclorobenceno (mg/L)",
    "Heptacloro + Heptacloroepóxido (mg/L)",
    "Metoxiclor (mg/L)",
    "Pentaclorofenol (mg/L)",
    "2,4-D (mg/L)",
    "Acrilamida (mg/L)",
    "Epiclorhidrina (mg/L)",
    "Cloruro de vinilo (mg/L)",
    "Benzo(a)pireno (mg/L)",
    "1,2-Dicloroetano (mg/L)",
    "Tetracloroeteno (mg/L)",
    "Monocloramina (mg/L)",
    "Tricloroeteno (mg/L)",
    "Tetracloruro de carbono (mg/L)",
    "Ftalato de di(2-etilhexilo) (mg/L)",
    "1,2-Diclorobenceno (mg/L)",
    "1,4-Diclorobenceno (mg/L)",
    "1,1-Dicloroeteno (mg/L)",
    "1,2-Dicloroeteno (mg/L)",
    "Diclorometano (mg/L)",
    "Ácido edético (EDTA) (mg/L)",
    "Etilbenceno (mg/L)",
    "Hexaclorobutadieno (mg/L)",
    "Ácido Nitrilotriacético (mg/L)",
    "Estireno (mg/L)",
    "Tolueno (mg/L)",
    "Xileno (mg/L)",
    "Atrazina (mg/L)",
    "Carbofurano (mg/L)",
    "Clorotoluron (mg/L)",
    "Cianazina (mg/L)",
    "2,4-DB (mg/L)",
    "1,2-Dibromo-3-Cloropropano (mg/L)",
    "1,2-Dibromoetano (mg/L)",
    "1,2-Dicloropropano (mg/L)",
    "1,3-Dicloropropeno (mg/L)",
    "Dicloroprop (mg/L)",
    "Dimetato (mg/L)",
    "Fenoprop (mg/L)",
    "Isoproturon (mg/L)",
    "MCPA (mg/L)",
    "Mecoprop (mg/L)",
    "Metolacloro (mg/L)",
    "Molinato (mg/L)",
    "Pendimetalina (mg/L)",
    "Simazina (mg/L)",
    "2,4,5-T (mg/L)",
    "Terbutilazina (mg/L)",
    "Trifluralina (mg/L)",
    "Cloropirifos (mg/L)",
    "Piriproxifeno (mg/L)",
    "Microcistin-LR (mg/L)",
    "Bromato (mg/L)",
    "Bromodiclorometano (mg/L)",
    "Bromoformo (mg/L)",
    "Hidrato de cloral (mg/L)",
    "Cloroformo (mg/L)",
    "Cloruro de cianógeno (mg/L)",
    "Dibromoacetonitrilo (mg/L)",
    "Dibromoclorometano (mg/L)",
    "Dicloroacetato (mg/L)",
    "Dicloroacetonitrilo (mg/L)",
    "Formaldehído (mg/L)",
    "Monocloroacetato (mg/L)",
    "Tricloroacetato (mg/L)",
    "2,4,6-Triclorofenol (mg/L)",
    "Malatión (mg/L)",
    "Bifenilos Policlorados (PCB) (mg/L)",

    # --- Radiactivos ---
    "Dosis de referencia total (mSv/año)",
    "Actividad global alfa (Bq/L)",
    "Actividad global beta (Bq/L)"
]

# ==========================================
# LÍMITES MINSA (DS N° 031-2010-SA)
# ==========================================
limites_minsa = {
    # Modelo
    "pH (Potencial de Hidrógeno)": (6.5, 8.5),
    "Turbidez (Unidades Nefelométricas - NTU)": 5.0,
    "Conductividad Eléctrica (µS/cm)": 1500.0,
    "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
    "Temperatura (°C)": 35.0,
    "Oxígeno Disuelto (mg/L)": None,  # Mínimo 4.0 se evalúa como invertido
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 3.0,
    "Nitratos (mg/L)": 50.0,
    "Coliformes Totales (NMP/100 mL)": 0.0,
    "Coliformes Termotolerantes (NMP/100 mL)": 0.0,
    "Plomo Total (mg/L)": 0.01,
    "Arsénico Total (mg/L)": 0.01,
    "Hierro Total (mg/L)": 0.3,
    "Sulfatos (mg/L)": 250.0,

    # Microbiológicos adicionales
    "Escherichia coli (UFC/100mL)": 0.0,
    "Bacterias Heterotróficas (UFC/mL)": 500.0,
    "Helmintos y protozoarios (N° org/L)": 0.0,
    "Virus (UFC/mL)": 0.0,
    "Organismos de vida libre (N° org/L)": 0.0,
    "Vibrio cholerae (Presencia/100mL)": None,  # Cualitativo

    # Organolépticos adicionales
    "Olor (Aceptable)": None,
    "Sabor (Aceptable)": None,
    "Color verdadero (UCV Pt/Co)": 15.0,
    "Cloruros (mg/L)": 250.0,
    "Dureza total (mg CaCO3/L)": 500.0,
    "Amoniaco (mg/L)": 1.5,
    "Manganeso Total (mg/L)": 0.4,
    "Aluminio (mg/L)": 0.2,
    "Cobre Total (mg/L)": 2.0,
    "Zinc (mg/L)": 3.0,
    "Sodio (mg/L)": 200.0,
    "Calcio (mg/L)": 200.0,
    "Magnesio (mg/L)": 150.0,
    "DQO (mg/L)": None,
    "Fenoles (mg/L)": None,
    "Fluoruros (mg/L)": 1.0,
    "Fósforo Total (mg/L)": None,
    "Materiales Flotantes (Ausencia)": None,
    "Nitritos (exposición corta) (mg NO2/L)": 3.0,
    "Nitritos (exposición larga) (mg NO2/L)": 0.2,
    "Aceites y Grasas (mg/L)": 0.5,

    # Inorgánicos
    "Antimonio (mg/L)": 0.02,
    "Bario (mg/L)": 0.7,
    "Berilio (mg/L)": None,
    "Boro (mg/L)": 0.5,
    "Cadmio Total (mg/L)": 0.003,
    "Cianuro total (mg/L)": 0.07,
    "Cianuro libre (mg/L)": None,
    "Cloro residual (mg/L)": 5.0,
    "Clorito (mg/L)": 0.7,
    "Clorato (mg/L)": 0.7,
    "Cromo total (mg/L)": 0.05,
    "Flúor (mg/L)": 1.0,
    "Mercurio Total (mg/L)": 0.001,
    "Níquel (mg/L)": 0.02,
    "Selenio (mg/L)": 0.01,
    "Molibdeno (mg/L)": 0.07,
    "Uranio (mg/L)": 0.015,

    # Orgánicos (solo los que tienen LMP definido en MINSA)
    "Trihalometanos totales (--)": 1.0,
    "Hidrocarburos de petróleo (C4-C10) (mg/L)": 0.01,
    "Alacloro (mg/L)": 0.02,
    "Aldicarb (mg/L)": 0.01,
    "Aldrín + Dieldrín (mg/L)": 0.0003,
    "Benceno (mg/L)": 0.01,
    "Clordano (total isómeros) (mg/L)": 0.0002,
    "DDT (total isómeros) (mg/L)": 0.001,
    "Endrin (mg/L)": 0.0006,
    "Lindano (Gamma HCH) (mg/L)": 0.002,
    "Hexaclorobenceno (mg/L)": 0.001,
    "Heptacloro + Heptacloroepóxido (mg/L)": 0.00003,
    "Metoxiclor (mg/L)": 0.02,
    "Pentaclorofenol (mg/L)": 0.009,
    "2,4-D (mg/L)": 0.03,
    "Acrilamida (mg/L)": 0.0005,
    "Epiclorhidrina (mg/L)": 0.0004,
    "Cloruro de vinilo (mg/L)": 0.0003,
    "Benzo(a)pireno (mg/L)": 0.0007,
    "1,2-Dicloroetano (mg/L)": 0.03,
    "Tetracloroeteno (mg/L)": 0.04,
    "Monocloramina (mg/L)": 3.0,
    "Tricloroeteno (mg/L)": 0.07,
    "Tetracloruro de carbono (mg/L)": 0.004,
    "Ftalato de di(2-etilhexilo) (mg/L)": 0.008,
    "1,2-Diclorobenceno (mg/L)": 1.0,
    "1,4-Diclorobenceno (mg/L)": 0.3,
    "1,1-Dicloroeteno (mg/L)": 0.03,
    "1,2-Dicloroeteno (mg/L)": 0.05,
    "Diclorometano (mg/L)": 0.02,
    "Ácido edético (EDTA) (mg/L)": 0.6,
    "Etilbenceno (mg/L)": 0.3,
    "Hexaclorobutadieno (mg/L)": 0.0006,
    "Ácido Nitrilotriacético (mg/L)": 0.2,
    "Estireno (mg/L)": 0.02,
    "Tolueno (mg/L)": 0.7,
    "Xileno (mg/L)": 0.5,
    "Atrazina (mg/L)": 0.002,
    "Carbofurano (mg/L)": 0.007,
    "Clorotoluron (mg/L)": 0.03,
    "Cianazina (mg/L)": 0.0006,
    "2,4-DB (mg/L)": 0.09,
    "1,2-Dibromo-3-Cloropropano (mg/L)": 0.001,
    "1,2-Dibromoetano (mg/L)": 0.0004,
    "1,2-Dicloropropano (mg/L)": 0.04,
    "1,3-Dicloropropeno (mg/L)": 0.02,
    "Dicloroprop (mg/L)": 0.1,
    "Dimetato (mg/L)": 0.006,
    "Fenoprop (mg/L)": 0.009,
    "Isoproturon (mg/L)": 0.009,
    "MCPA (mg/L)": 0.002,
    "Mecoprop (mg/L)": 0.01,
    "Metolacloro (mg/L)": 0.01,
    "Molinato (mg/L)": 0.006,
    "Pendimetalina (mg/L)": 0.02,
    "Simazina (mg/L)": 0.002,
    "2,4,5-T (mg/L)": 0.009,
    "Terbutilazina (mg/L)": 0.007,
    "Trifluralina (mg/L)": 0.02,
    "Cloropirifos (mg/L)": 0.03,
    "Piriproxifeno (mg/L)": 0.3,
    "Microcistin-LR (mg/L)": 0.001,
    "Bromato (mg/L)": 0.01,
    "Bromodiclorometano (mg/L)": 0.06,
    "Bromoformo (mg/L)": 0.1,
    "Hidrato de cloral (mg/L)": 0.01,
    "Cloroformo (mg/L)": 0.2,
    "Cloruro de cianógeno (mg/L)": 0.07,
    "Dibromoacetonitrilo (mg/L)": 0.1,
    "Dibromoclorometano (mg/L)": 0.05,
    "Dicloroacetato (mg/L)": 0.02,
    "Dicloroacetonitrilo (mg/L)": 0.9,
    "Formaldehído (mg/L)": 0.02,
    "Monocloroacetato (mg/L)": 0.2,
    "Tricloroacetato (mg/L)": 0.2,
    "2,4,6-Triclorofenol (mg/L)": 0.2,
    "Malatión (mg/L)": None,
    "Bifenilos Policlorados (PCB) (mg/L)": None,

    # Radiactivos
    "Dosis de referencia total (mSv/año)": 0.1,
    "Actividad global alfa (Bq/L)": 0.5,
    "Actividad global beta (Bq/L)": 1.0
}

# ==========================================
# PARÁMETROS CON LÍMITE MÍNIMO (INVERTIDO) – MINSA
# ==========================================
INVERTIDOS_MINSA = {"Oxígeno Disuelto (mg/L)"}

# ==========================================
# LÍMITES ECA CATEGORÍA 1 (A1, A2, A3)
# ==========================================
limites_eca = {
    "A1": {},
    "A2": {},
    "A3": {}
}

# ----- A1 -----
limites_eca["A1"] = {
    "pH (Potencial de Hidrógeno)": (6.5, 8.5),
    "Turbidez (Unidades Nefelométricas - NTU)": 5.0,
    "Conductividad Eléctrica (µS/cm)": 1500.0,
    "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
    "Temperatura (°C)": 25.0,  # Δ3 respecto al promedio, se usa 25°C como referencia
    "Oxígeno Disuelto (mg/L)": 6.0,   # mínimo
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 3.0,
    "Nitratos (mg/L)": 50.0,
    "Coliformes Totales (NMP/100 mL)": 50.0,
    "Coliformes Termotolerantes (NMP/100 mL)": 20.0,
    "Plomo Total (mg/L)": 0.01,
    "Arsénico Total (mg/L)": 0.01,
    "Hierro Total (mg/L)": 0.3,
    "Sulfatos (mg/L)": 250.0,
    # adicionales
    "Escherichia coli (UFC/100mL)": 0.0,
    "Helmintos y protozoarios (N° org/L)": 0.0,
    "Vibrio cholerae (Presencia/100mL)": 0.0,
    "Organismos de vida libre (N° org/L)": 0.0,
    "Color verdadero (UCV Pt/Co)": 15.0,
    "Cloruros (mg/L)": 250.0,
    "Dureza total (mg CaCO3/L)": 500.0,
    "Amoniaco (mg/L)": 1.5,
    "Manganeso Total (mg/L)": 0.4,
    "Aluminio (mg/L)": 0.9,
    "Cobre Total (mg/L)": 2.0,
    "Zinc (mg/L)": 3.0,
    "Sodio (mg/L)": None,
    "Calcio (mg/L)": None,
    "Magnesio (mg/L)": None,
    "DQO (mg/L)": 10.0,
    "Fenoles (mg/L)": 0.003,
    "Fluoruros (mg/L)": 1.5,
    "Fósforo Total (mg/L)": 0.1,
    "Materiales Flotantes (Ausencia)": 0.0,
    "Nitritos (exposición corta) (mg NO2/L)": 3.0,
    "Aceites y Grasas (mg/L)": 0.5,
    "Antimonio (mg/L)": 0.02,
    "Arsénico Total (mg/L)": 0.01,
    "Bario (mg/L)": 0.7,
    "Berilio (mg/L)": 0.012,
    "Boro (mg/L)": 2.4,
    "Cadmio Total (mg/L)": 0.003,
    "Cianuro total (mg/L)": 0.07,
    "Cianuro libre (mg/L)": None,
    "Cloro residual (mg/L)": None,
    "Clorito (mg/L)": None,
    "Clorato (mg/L)": None,
    "Cromo total (mg/L)": 0.05,
    "Flúor (mg/L)": 1.5,
    "Mercurio Total (mg/L)": 0.001,
    "Níquel (mg/L)": 0.07,
    "Selenio (mg/L)": 0.04,
    "Molibdeno (mg/L)": 0.07,
    "Uranio (mg/L)": 0.02,
    # orgánicos A1
    "Trihalometanos totales (--)": 1.0,
    "Hidrocarburos de petróleo (C4-C10) (mg/L)": 0.01,
    "Aldicarb (mg/L)": 0.01,
    "Aldrín + Dieldrín (mg/L)": 0.00003,
    "Benceno (mg/L)": 0.01,
    "Clordano (total isómeros) (mg/L)": 0.0002,
    "DDT (total isómeros) (mg/L)": 0.001,
    "Endrin (mg/L)": 0.0006,
    "Lindano (Gamma HCH) (mg/L)": 0.002,
    "Heptacloro + Heptacloroepóxido (mg/L)": 0.00003,
    "Pentaclorofenol (mg/L)": 0.009,
    "Benzo(a)pireno (mg/L)": 0.0007,
    "1,2-Dicloroetano (mg/L)": 0.03,
    "Tetracloroeteno (mg/L)": 0.04,
    "Tricloroeteno (mg/L)": 0.07,
    "Tetracloruro de carbono (mg/L)": 0.004,
    "Etilbenceno (mg/L)": 0.3,
    "Hexaclorobutadieno (mg/L)": 0.0006,
    "Tolueno (mg/L)": 0.7,
    "Xileno (mg/L)": 0.5,
    "Microcistin-LR (mg/L)": 0.001,
    "Bromodiclorometano (mg/L)": 0.06,
    "Bromoformo (mg/L)": 0.1,
    "Cloroformo (mg/L)": 0.3,
    "Dibromoclorometano (mg/L)": 0.1,
    "Malatión (mg/L)": 0.19,
    "Bifenilos Policlorados (PCB) (mg/L)": 0.0005
}

# ----- A2 -----
limites_eca["A2"] = {
    "pH (Potencial de Hidrógeno)": (5.5, 9.0),
    "Turbidez (Unidades Nefelométricas - NTU)": 100.0,
    "Conductividad Eléctrica (µS/cm)": 1600.0,
    "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
    "Temperatura (°C)": 25.0,
    "Oxígeno Disuelto (mg/L)": 5.0,
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 5.0,
    "Nitratos (mg/L)": 50.0,
    "Coliformes Totales (NMP/100 mL)": None,
    "Coliformes Termotolerantes (NMP/100 mL)": 2000.0,
    "Plomo Total (mg/L)": 0.05,
    "Arsénico Total (mg/L)": 0.01,
    "Hierro Total (mg/L)": 1.0,
    "Sulfatos (mg/L)": 500.0,
    "Escherichia coli (UFC/100mL)": None,
    "Helmintos y protozoarios (N° org/L)": None,
    "Vibrio cholerae (Presencia/100mL)": 0.0,
    "Organismos de vida libre (N° org/L)": 5e6,
    "Color verdadero (UCV Pt/Co)": 100.0,
    "Cloruros (mg/L)": 250.0,
    "Dureza total (mg CaCO3/L)": None,
    "Amoniaco (mg/L)": 1.5,
    "Manganeso Total (mg/L)": 0.4,
    "Aluminio (mg/L)": 5.0,
    "Cobre Total (mg/L)": 2.0,
    "Zinc (mg/L)": 5.0,
    "DQO (mg/L)": 20.0,
    "Fenoles (mg/L)": None,
    "Fluoruros (mg/L)": None,
    "Fósforo Total (mg/L)": 0.15,
    "Materiales Flotantes (Ausencia)": 0.0,
    "Nitritos (exposición corta) (mg NO2/L)": 3.0,
    "Aceites y Grasas (mg/L)": 1.7,
    "Antimonio (mg/L)": 0.02,
    "Berilio (mg/L)": 0.04,
    "Boro (mg/L)": 2.4,
    "Cadmio Total (mg/L)": 0.005,
    "Cianuro libre (mg/L)": 0.2,
    "Cromo total (mg/L)": 0.05,
    "Mercurio Total (mg/L)": 0.002,
    "Selenio (mg/L)": 0.04,
    "Uranio (mg/L)": 0.02,
    "Aldicarb (mg/L)": 0.01,
    "Aldrín + Dieldrín (mg/L)": 0.00003,
    "Benceno (mg/L)": 0.01,
    "Clordano (total isómeros) (mg/L)": 0.0002,
    "DDT (total isómeros) (mg/L)": 0.001,
    "Endrin (mg/L)": 0.0006,
    "Lindano (Gamma HCH) (mg/L)": 0.002,
    "Heptacloro + Heptacloroepóxido (mg/L)": 0.00003,
    "Pentaclorofenol (mg/L)": 0.009,
    "Benzo(a)pireno (mg/L)": 0.0007,
    "1,2-Dicloroetano (mg/L)": 0.03,
    "Tetracloroeteno (mg/L)": None,
    "Tricloroeteno (mg/L)": 0.07,
    "Tetracloruro de carbono (mg/L)": 0.004,
    "Etilbenceno (mg/L)": 0.3,
    "Hexaclorobutadieno (mg/L)": 0.0006,
    "Tolueno (mg/L)": 0.7,
    "Xileno (mg/L)": 0.5,
    "Microcistin-LR (mg/L)": 0.001,
    "Malatión (mg/L)": 0.0001,
    "Bifenilos Policlorados (PCB) (mg/L)": 0.0005
}

# ----- A3 -----
limites_eca["A3"] = {
    "pH (Potencial de Hidrógeno)": (5.5, 9.0),
    "Turbidez (Unidades Nefelométricas - NTU)": 100.0,  # (ECA A3: 100 UNT, en el doc original es 100)
    "Conductividad Eléctrica (µS/cm)": None,
    "Sólidos Totales Disueltos - TDS (mg/L)": 1500.0,
    "Temperatura (°C)": 25.0,
    "Oxígeno Disuelto (mg/L)": 4.0,
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 10.0,
    "Nitratos (mg/L)": None,
    "Coliformes Totales (NMP/100 mL)": None,
    "Coliformes Termotolerantes (NMP/100 mL)": 20000.0,
    "Plomo Total (mg/L)": 0.05,
    "Arsénico Total (mg/L)": 0.15,
    "Hierro Total (mg/L)": 5.0,
    "Sulfatos (mg/L)": None,
    "Vibrio cholerae (Presencia/100mL)": 0.0,
    "Organismos de vida libre (N° org/L)": 5e6,
    "Color verdadero (UCV Pt/Co)": None,
    "Cloruros (mg/L)": None,
    "Amoniaco (mg/L)": None,
    "Manganeso Total (mg/L)": 0.5,
    "Aluminio (mg/L)": 5.0,
    "Cobre Total (mg/L)": 2.0,
    "Zinc (mg/L)": 5.0,
    "DQO (mg/L)": 30.0,
    "Fósforo Total (mg/L)": 0.15,
    "Materiales Flotantes (Ausencia)": 0.0,
    "Aceites y Grasas (mg/L)": 1.7,
    "Berilio (mg/L)": 0.1,
    "Boro (mg/L)": 2.4,
    "Cadmio Total (mg/L)": 0.01,
    "Cianuro libre (mg/L)": 0.2,
    "Cromo total (mg/L)": 0.05,
    "Mercurio Total (mg/L)": 0.002,
    "Selenio (mg/L)": 0.05,
    "Uranio (mg/L)": 0.02
}

# ==========================================
# PARÁMETROS CON LÍMITE MÍNIMO (INVERTIDO) – ECA
# ==========================================
INVERTIDOS_ECA = {"Oxígeno Disuelto (mg/L)"}  # Se aplica a todas las categorías

# ==========================================
# INTERFAZ DE USUARIO
# ==========================================
st.title("💧 Sistema de Evaluación de Calidad de Agua")
st.markdown("Evalúa la potabilidad y requerimientos de tratamiento basados en **MINSA**, **ECA (Categoría 1)** y un **Modelo de Inteligencia Artificial**.")

st.sidebar.header("Categoría ECA a Evaluar")
eca_categoria = st.sidebar.selectbox("Seleccione la subcategoría del cuerpo receptor:", ["A1", "A2", "A3"],
                                     help="A1: Desinfección, A2: Tratamiento Convencional, A3: Tratamiento Avanzado")

st.sidebar.header("Ingreso de Parámetros")
st.sidebar.write("Ingrese los valores de la muestra:")

# Crear inputs en el sidebar (se usa un expander para no saturar)
valores_usuario = {}
with st.sidebar.expander("Parámetros (click para expandir)", expanded=True):
    for param in PARAMETROS_NOMBRES:
        # Valor por defecto 0.0, paso adecuado
        if "Coliformes" in param or "Escherichia" in param:
            step = 1.0
        else:
            step = 0.1
        valores_usuario[param] = st.number_input(param, value=0.0, step=step, format="%.4f")

# Botón de evaluación
if st.sidebar.button("Evaluar Calidad de Agua"):

    # --- 1. EVALUACIÓN NORMATIVA MINSA ---
    cumplen_minsa = []
    no_cumplen_minsa = []

    for param in PARAMETROS_NOMBRES:
        valor = valores_usuario[param]
        limite = limites_minsa.get(param)

        if limite is None:
            continue  # Sin límite, no se evalúa

        if param in INVERTIDOS_MINSA:
            if valor >= limite:
                cumplen_minsa.append(param)
            else:
                no_cumplen_minsa.append(param)
        elif isinstance(limite, tuple):
            if limite[0] <= valor <= limite[1]:
                cumplen_minsa.append(param)
            else:
                no_cumplen_minsa.append(param)
        else:  # Límite máximo
            if valor <= limite:
                cumplen_minsa.append(param)
            else:
                no_cumplen_minsa.append(param)

    # --- 2. EVALUACIÓN NORMATIVA ECA ---
    cumplen_eca = []
    no_cumplen_eca = []
    limite_actual = limites_eca[eca_categoria]

    for param in PARAMETROS_NOMBRES:
        valor = valores_usuario[param]
        limite = limite_actual.get(param)

        if limite is None:
            continue

        if param in INVERTIDOS_ECA:
            if valor >= limite:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)
        elif isinstance(limite, tuple):
            if limite[0] <= valor <= limite[1]:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)
        else:
            if valor <= limite:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)

    # --- 3. PREDICCIÓN DEL MODELO (IA) ---
    if modelo is not None:
        # Las 14 características que espera el modelo (en el mismo orden)
        FEATURES_MODELO = [
            "pH (Potencial de Hidrógeno)",
            "Turbidez (Unidades Nefelométricas - NTU)",
            "Conductividad Eléctrica (µS/cm)",
            "Sólidos Totales Disueltos - TDS (mg/L)",
            "Temperatura (°C)",
            "Oxígeno Disuelto (mg/L)",
            "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)",
            "Nitratos (mg/L)",
            "Coliformes Totales (NMP/100 mL)",
            "Coliformes Termotolerantes (NMP/100 mL)",
            "Plomo Total (mg/L)",
            "Arsénico Total (mg/L)",
            "Hierro Total (mg/L)",
            "Sulfatos (mg/L)"
        ]
        input_data = np.array([[valores_usuario[p] for p in FEATURES_MODELO]])
        prediccion = modelo.predict(input_data)[0]
        resultado_ia = "Potable" if prediccion == 1 else "No Potable"
    else:
        resultado_ia = "Modelo no disponible (verifique el archivo .pkl)"

    # --- 4. MOSTRAR RESULTADOS ---
    st.header("📊 Resultados de la Evaluación")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evaluación según MINSA (DS N° 031-2010-SA)")
        if len(no_cumplen_minsa) == 0:
            st.success("✅ El agua CUMPLE con los estándares de potabilidad del MINSA (parámetros ingresados).")
        else:
            st.error(f"❌ {len(no_cumplen_minsa)} parámetros NO CUMPLEN con el MINSA.")

        with st.expander("Detalle MINSA (parámetros evaluados)", expanded=True):
            if cumplen_minsa:
                st.write("**Cumplen:**")
                for p in cumplen_minsa:
                    st.write(f"- {p}: {valores_usuario[p]}")
            if no_cumplen_minsa:
                st.write("**No cumplen:**")
                for p in no_cumplen_minsa:
                    st.write(f"- {p}: {valores_usuario[p]} (Límite: {limites_minsa[p]})")

    with col2:
        st.subheader(f"Evaluación según ECA – Subcategoría {eca_categoria}")
        if len(no_cumplen_eca) == 0:
            st.success(f"✅ El agua CUMPLE con los estándares ECA {eca_categoria}.")
        else:
            st.error(f"❌ {len(no_cumplen_eca)} parámetros NO CUMPLEN con ECA {eca_categoria}.")

        with st.expander(f"Detalle ECA {eca_categoria} (parámetros evaluados)", expanded=True):
            if cumplen_eca:
                st.write("**Cumplen:**")
                for p in cumplen_eca:
                    st.write(f"- {p}: {valores_usuario[p]}")
            if no_cumplen_eca:
                st.write("**No cumplen:**")
                for p in no_cumplen_eca:
                    st.write(f"- {p}: {valores_usuario[p]} (Límite: {limite_actual[p]})")

    st.markdown("---")

    # --- 5. MODELO IA Y TRATAMIENTO SUGERIDO ---
    st.header("🤖 Predicción de Inteligencia Artificial")
    if resultado_ia == "Potable":
        st.success(f"El modelo predictivo clasifica esta muestra como: **{resultado_ia}**")
    elif resultado_ia == "No Potable":
        st.warning(f"El modelo predictivo clasifica esta muestra como: **{resultado_ia}**")
    else:
        st.info(resultado_ia)

    st.header("🛠️ Sugerencia de Tratamiento Requerido")
    if len(no_cumplen_minsa) == 0:
        st.info("El agua ya cumple con los estándares MINSA. Solo se requiere asegurar la cloración residual en la red de distribución.")
    elif len(no_cumplen_eca) == 0:
        if eca_categoria == "A1":
            st.info("Fuente de agua categoría A1. Tratamiento sugerido: **Desinfección simple** (cloración, ozonización).")
        elif eca_categoria == "A2":
            st.info("Fuente de agua categoría A2. Tratamiento sugerido: **Tratamiento convencional** (coagulación, floculación, decantación, filtración, desinfección).")
        elif eca_categoria == "A3":
            st.info("Fuente de agua categoría A3. Tratamiento sugerido: **Tratamiento avanzado** (procesos convencionales + ósmosis inversa, carbón activado, oxidación avanzada).")
    else:
        st.error(f"El agua excede los límites de la categoría ECA {eca_categoria}. Se requieren tratamientos específicos para remover los parámetros infractores mostrados arriba, o buscar una fuente alternativa.")

else:
    st.info("👈 Ingrese los valores en el panel lateral y haga clic en 'Evaluar Calidad de Agua'.")
