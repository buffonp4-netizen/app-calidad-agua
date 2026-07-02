import streamlit as st
import pandas as pd
import numpy as np
import time

# =========================================================================
# CONFIGURACIÓN Y ESTADO
# =========================================================================
st.set_page_config(page_title="Sistema Híbrido Calidad Agua", page_icon="💧", layout="wide")

# =========================================================================
# ZONA DE DATOS
# =========================================================================

# Campos del modelo
campos_modelo = {
    'pH': 7.0, 'Conductividad': 500.0, 'Temperatura': 20.0, 'Oxígeno Disuelto': 5.0,
    'DBO5': 2.0, 'Coliformes Totales': 0.0, 'Aceites y Grasas': 0.0,
    'Arsénico': 0.0, 'Plomo': 0.0, 'Cobre': 0.0, 'Manganeso': 0.0,
    'Calcio': 0.0, 'Magnesio': 0.0, 'Dureza total': 0.0
}

# Diccionario normativo completo (MINSA + ECA Categoría 1 A1/A2/A3)
NORMATIVA_COMPLETA = {
    # ===== 1. Microbiológicos =====
    "Coliformes Totales":            {'minsa': 0,     'eca_a1': 50,    'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Escherichia coli":              {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/100mL',   'categoria': '1. Microbiológicos'},
    "Coliformes Termotolerantes":    {'minsa': 0,     'eca_a1': 20,    'eca_a2': 2000,  'eca_a3': 20000,  'invertido': False, 'unidad': 'NMP/100mL',   'categoria': '1. Microbiológicos'},
    "Bacterias Heterotróficas":      {'minsa': 500,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Helmintos y protozoarios":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Virus":                         {'minsa': 0,     'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'UFC/mL',      'categoria': '1. Microbiológicos'},
    "Organismos de vida libre":      {'minsa': 0,     'eca_a1': 0,     'eca_a2': 5e6,   'eca_a3': 5e6,    'invertido': False, 'unidad': 'N° org/L',    'categoria': '1. Microbiológicos'},
    "Vibrio cholerae":               {'minsa': None,  'eca_a1': 0,     'eca_a2': 0,     'eca_a3': 0,      'invertido': False, 'unidad': 'Presencia/100mL','categoria': '1. Microbiológicos'},

    # ===== 2. Organolépticos =====
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

    # ===== 3. Inorgánicos =====
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

    # ===== 4. Orgánicos =====
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

    # ===== 5. Radiactivos =====
    "Dosis de referencia total":     {'minsa': 0.1,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'mSv/año',    'categoria': '5. Radiactivos'},
    "Actividad global alfa":         {'minsa': 0.5,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
    "Actividad global beta":         {'minsa': 1.0,   'eca_a1': None,  'eca_a2': None,  'eca_a3': None,   'invertido': False, 'unidad': 'Bq/L',       'categoria': '5. Radiactivos'},
}

# =========================================================================
# FUNCIÓN DE INICIALIZACIÓN DE ESTADO
# =========================================================================
def inicializar_estado():
    for k in campos_modelo:
        if f"tog_{k}" not in st.session_state:
            st.session_state[f"tog_{k}"] = False
        if f"val_{k}" not in st.session_state:
            st.session_state[f"val_{k}"] = 0.0

    for k in NORMATIVA_COMPLETA.keys():
        if f"tog_norm_{k}" not in st.session_state:
            st.session_state[f"tog_norm_{k}"] = False
        if f"val_norm_{k}" not in st.session_state:
            st.session_state[f"val_norm_{k}"] = 0.0

    if "resultados_minsa" not in st.session_state:
        st.session_state["resultados_minsa"] = None
    if "detalles_eca" not in st.session_state:
        st.session_state["detalles_eca"] = None
    if "peor_categoria_eca" not in st.session_state:
        st.session_state["peor_categoria_eca"] = None

inicializar_estado()

# =========================================================================
# FUNCIONES AUXILIARES
# =========================================================================
def render_param(label, key, default_val=0.0):
    col1, col2 = st.columns([0.1, 1])
    with col1:
        st.toggle(" ", key=f"tog_{key}")
    with col2:
        is_active = st.session_state[f"tog_{key}"]
        val = st.session_state.get(f"val_{key}", default_val)
        st.number_input(label, value=val, disabled=not is_active, format="%.4f", key=f"val_{key}")

# =========================================================================
# INTERFAZ
# =========================================================================
st.title("💧 Sistema Híbrido de Diagnóstico")
busqueda = st.text_input("🔍 Buscar parámetro:", placeholder="Escribe para filtrar...")

tab1, tab2, tab3 = st.tabs(["📊 1. Modelo Entrenado", "⚖️ 2. Normativa Completa", "🔬 3. Diagnóstico Final"])

with tab1:
    st.subheader("Configuración de Parámetros del Modelo")
    c1, c2 = st.columns(2)
    if c1.button("✅ Activar todos", key="btn_act_model"):
        for k in campos_modelo:
            st.session_state[f"tog_{k}"] = True
    if c2.button("❌ Desactivar todos", key="btn_des_model"):
        for k in campos_modelo:
            st.session_state[f"tog_{k}"] = False

    for k, v in campos_modelo.items():
        if busqueda.lower() in k.lower():
            render_param(k, k, v)

with tab2:
    st.subheader("Configuración Normativa")
    categorias = sorted(list(set(info['categoria'] for info in NORMATIVA_COMPLETA.values())))
    for cat in categorias:
        with st.expander(f"📁 {cat}", expanded=True):
            for param, info in NORMATIVA_COMPLETA.items():
                if info['categoria'] == cat and (busqueda.lower() in param.lower() or busqueda == ""):
                    render_param(f"{param} ({info['unidad']})", f"norm_{param}", 0.0)

# --- TAB 3: Diagnóstico Final (COMPLETO) ---
with tab3:
    st.subheader("Resultados del Análisis")

    active_model_keys = [k for k in campos_modelo.keys() if st.session_state.get(f"tog_{k}", False)]
    active_norm_keys = [k for k in NORMATIVA_COMPLETA.keys() if st.session_state.get(f"tog_norm_{k}", False)]

    st.markdown("### 🤖 Diagnóstico: Modelo Entrenado")
    st.metric("Parámetros seleccionados", len(active_model_keys))

    if st.button("🚀 Ejecutar Diagnóstico", use_container_width=True):
        with st.spinner("Procesando datos con IA..."):
            # 1. Validación MINSA
            incumplimientos = []
            for param in active_norm_keys:
                valor = st.session_state.get(f"val_norm_{param}", 0.0)
                info = NORMATIVA_COMPLETA[param]
                minsa_lim = info['minsa']
                if minsa_lim is None:
                    continue
                if isinstance(minsa_lim, tuple):
                    if not (minsa_lim[0] <= valor <= minsa_lim[1]):
                        incumplimientos.append((param, valor, f"Rango {minsa_lim[0]} - {minsa_lim[1]}"))
                elif info.get('invertido', False):
                    if valor < minsa_lim:
                        incumplimientos.append((param, valor, f"Mínimo: {minsa_lim}"))
                else:
                    if valor > minsa_lim:
                        incumplimientos.append((param, valor, f"Máximo: {minsa_lim}"))

            # 2. Clasificación ECA
            peor_cat = "A1"
            detalles_eca = []
            orden_eca = {"A1": 1, "A2": 2, "A3": 3, "EXCEDE A3": 4}
            for param in active_norm_keys:
                valor = st.session_state.get(f"val_norm_{param}", 0.0)
                info = NORMATIVA_COMPLETA[param]
                cat = None
                invertido = info.get('invertido', False)
                for subcat in ['eca_a1', 'eca_a2', 'eca_a3']:
                    lim = info.get(subcat)
                    if lim is None:
                        continue
                    if isinstance(lim, tuple):
                        if lim[0] <= valor <= lim[1]:
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
                if orden_eca[cat] > orden_eca[peor_cat]:
                    peor_cat = cat

            st.session_state["resultados_minsa"] = incumplimientos
            st.session_state["detalles_eca"] = detalles_eca
            st.session_state["peor_categoria_eca"] = peor_cat

    # Mostrar resultados si ya se ejecutó
    if st.session_state["resultados_minsa"] is not None:
        st.markdown("---")
        st.markdown("### 🏛️ Resultados de Evaluación")

        if st.session_state["resultados_minsa"]:
            st.error(f"⚠️ {len(st.session_state['resultados_minsa'])} parámetros NO cumplen MINSA")
            st.table(pd.DataFrame(st.session_state["resultados_minsa"], columns=['Parámetro', 'Valor', 'Límite']))
        else:
            st.success("✅ Todo CUMPLE con la normativa MINSA.")

        st.markdown(f"#### Peor nivel detectado: `{st.session_state['peor_categoria_eca']}`")
        if st.session_state['peor_categoria_eca'] == "A1":
            st.info("🟢 A1 – Desinfección simple (cloración, ozonización)")
        elif st.session_state['peor_categoria_eca'] == "A2":
            st.warning("🟠 A2 – Tratamiento convencional (coagulación, floculación, filtración, desinfección)")
        elif st.session_state['peor_categoria_eca'] == "A3":
            st.error("🔴 A3 – Tratamiento avanzado (ósmosis inversa, carbón activado, oxidación avanzada)")
        else:
            st.error("⚫ EXCEDE A3 – No apta para potabilización convencional. Descarte o tratamiento especial.")

        with st.expander("Ver clasificación ECA detallada"):
            st.dataframe(pd.DataFrame(st.session_state["detalles_eca"]))
