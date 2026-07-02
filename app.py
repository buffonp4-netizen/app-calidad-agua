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
        # Carga el modelo desde el archivo pickle proporcionado
        return pickle.load(open('modelo_agua.pkl', 'rb'))
    except Exception as e:
        return None

modelo = load_model()

# ==========================================
# DICCIONARIOS DE LÍMITES NORMATIVOS
# ==========================================
# Nota: Los límites están simplificados a valores máximos permitidos (o rangos para el pH)
# basados en DS N° 031-2010-SA (MINSA) y DS N° 004-2017-MINAM (ECA Categoría 1).

# Parámetros (Nombres Completos)
PARAMETROS_NOMBRES = [
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

# Límites MINSA (Agua potable)
limites_minsa = {
    "pH (Potencial de Hidrógeno)": (6.5, 8.5), # Rango
    "Turbidez (Unidades Nefelométricas - NTU)": 5.0,
    "Conductividad Eléctrica (µS/cm)": 1500.0,
    "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
    "Temperatura (°C)": 35.0, # Valor referencial
    "Oxígeno Disuelto (mg/L)": 0.0, # No aplica directamente en red, se asume sin límite restrictivo
    "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 0.0, # No aplica
    "Nitratos (mg/L)": 50.0,
    "Coliformes Totales (NMP/100 mL)": 0.0, # Ausencia
    "Coliformes Termotolerantes (NMP/100 mL)": 0.0, # Ausencia
    "Plomo Total (mg/L)": 0.01,
    "Arsénico Total (mg/L)": 0.01,
    "Hierro Total (mg/L)": 0.3,
    "Sulfatos (mg/L)": 250.0
}

# Límites ECA Agua Categoría 1 (A1, A2, A3)
limites_eca = {
    "A1": {
        "pH (Potencial de Hidrógeno)": (6.5, 8.5),
        "Turbidez (Unidades Nefelométricas - NTU)": 5.0,
        "Conductividad Eléctrica (µS/cm)": 1500.0,
        "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
        "Temperatura (°C)": 35.0,
        "Oxígeno Disuelto (mg/L)": 6.0, # Mínimo (Lógica invertida en la evaluación)
        "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 3.0,
        "Nitratos (mg/L)": 50.0,
        "Coliformes Totales (NMP/100 mL)": 50.0,
        "Coliformes Termotolerantes (NMP/100 mL)": 20.0,
        "Plomo Total (mg/L)": 0.01,
        "Arsénico Total (mg/L)": 0.01,
        "Hierro Total (mg/L)": 0.3,
        "Sulfatos (mg/L)": 250.0
    },
    "A2": {
        "pH (Potencial de Hidrógeno)": (5.5, 9.0),
        "Turbidez (Unidades Nefelométricas - NTU)": 100.0,
        "Conductividad Eléctrica (µS/cm)": 1500.0,
        "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
        "Temperatura (°C)": 35.0,
        "Oxígeno Disuelto (mg/L)": 5.0, # Mínimo
        "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 5.0,
        "Nitratos (mg/L)": 50.0,
        "Coliformes Totales (NMP/100 mL)": 3000.0,
        "Coliformes Termotolerantes (NMP/100 mL)": 2000.0,
        "Plomo Total (mg/L)": 0.05,
        "Arsénico Total (mg/L)": 0.05,
        "Hierro Total (mg/L)": 1.0,
        "Sulfatos (mg/L)": 500.0
    },
    "A3": {
        "pH (Potencial de Hidrógeno)": (5.5, 9.0),
        "Turbidez (Unidades Nefelométricas - NTU)": 1000.0, # Relajado para A3
        "Conductividad Eléctrica (µS/cm)": 1500.0,
        "Sólidos Totales Disueltos - TDS (mg/L)": 1000.0,
        "Temperatura (°C)": 35.0,
        "Oxígeno Disuelto (mg/L)": 4.0, # Mínimo
        "Demanda Bioquímica de Oxígeno - DBO5 (mg/L)": 15.0,
        "Nitratos (mg/L)": 50.0,
        "Coliformes Totales (NMP/100 mL)": 50000.0,
        "Coliformes Termotolerantes (NMP/100 mL)": 20000.0,
        "Plomo Total (mg/L)": 0.05,
        "Arsénico Total (mg/L)": 0.1,
        "Hierro Total (mg/L)": 5.0,
        "Sulfatos (mg/L)": 500.0
    }
}

# ==========================================
# INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("💧 Sistema de Evaluación de Calidad de Agua")
st.markdown("Evalúa la potabilidad y los requerimientos de tratamiento basados en **MINSA**, **ECA (Categoría 1)** y un **Modelo de Inteligencia Artificial**.")

st.sidebar.header("Categoría ECA a Evaluar")
eca_categoria = st.sidebar.selectbox("Seleccione la subcategoría del cuerpo receptor:", ["A1", "A2", "A3"], 
                                     help="A1: Desinfección, A2: Tratamiento Convencional, A3: Tratamiento Avanzado")

st.sidebar.header("Ingreso de Parámetros")
st.sidebar.write("Por favor, ingrese los valores de la muestra:")

# Crear inputs dinámicamente en el sidebar
valores_usuario = {}
for param in PARAMETROS_NOMBRES:
    valores_usuario[param] = st.sidebar.number_input(param, value=0.0, step=0.1)

# Botón de evaluación
if st.sidebar.button("Evaluar Calidad de Agua"):
    
    # --- 1. EVALUACIÓN NORMATIVA (MINSA y ECA) ---
    cumplen_minsa = []
    no_cumplen_minsa = []
    
    cumplen_eca = []
    no_cumplen_eca = []

    for param in PARAMETROS_NOMBRES:
        valor = valores_usuario[param]
        
        # Evaluación MINSA
        limite_m = limites_minsa[param]
        if isinstance(limite_m, tuple): # Caso especial: rangos (como pH)
            if limite_m[0] <= valor <= limite_m[1]:
                cumplen_minsa.append(param)
            else:
                no_cumplen_minsa.append(param)
        elif param == "Oxígeno Disuelto (mg/L)": # Caso especial: mínimo requerido
            pass # No se penaliza en MINSA
        else: # Límite máximo
            if valor <= limite_m:
                cumplen_minsa.append(param)
            else:
                no_cumplen_minsa.append(param)

        # Evaluación ECA
        limite_e = limites_eca[eca_categoria][param]
        if isinstance(limite_e, tuple): # Rango (pH)
            if limite_e[0] <= valor <= limite_e[1]:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)
        elif param == "Oxígeno Disuelto (mg/L)": # Mínimo requerido
            if valor >= limite_e:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)
        else: # Máximo permitido
            if valor <= limite_e:
                cumplen_eca.append(param)
            else:
                no_cumplen_eca.append(param)

    # --- 2. PREDICCIÓN DEL MODELO (IA) ---
    if modelo is not None:
        # Extraer solo los valores numéricos en el orden correcto para el modelo
        input_data = np.array([[valores_usuario[p] for p in PARAMETROS_NOMBRES]])
        prediccion = modelo.predict(input_data)[0]
        resultado_ia = "Potable" if prediccion == 1 else "No Potable"
    else:
        resultado_ia = "Modelo no disponible (Verifique el archivo .pkl)"

    # --- 3. MOSTRAR RESULTADOS ---
    st.header("📊 Resultados de la Evaluación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluación según MINSA (DS N° 031-2010-SA)")
        if len(no_cumplen_minsa) == 0:
            st.success("✅ El agua es considerada POTABLE según los parámetros ingresados.")
        else:
            st.error("❌ El agua NO ES POTABLE según los estándares del MINSA.")
            
        # NUEVO: Mostrar parámetros que SÍ cumplieron con MINSA
        with st.expander("Ver parámetros que SÍ cumplieron (MINSA)", expanded=True):
            if cumplen_minsa:
                for p in cumplen_minsa:
                    st.write(f"- {p}: **{valores_usuario[p]}** (Límite: {limites_minsa[p]})")
            else:
                st.write("Ningún parámetro cumplió.")

    with col2:
        st.subheader(f"Evaluación según ECA Agua - Subcategoría {eca_categoria}")
        if len(no_cumplen_eca) == 0:
            st.success(f"✅ El agua de la fuente CUMPLE con los estándares para la categoría {eca_categoria}.")
        else:
            st.error(f"❌ El agua NO CUMPLE con la normativa ECA {eca_categoria}.")
            
        # NUEVO: Mostrar parámetros que NO cumplieron con ECA
        with st.expander(f"Ver parámetros que NO cumplieron (ECA {eca_categoria})", expanded=True):
            if no_cumplen_eca:
                for p in no_cumplen_eca:
                    st.write(f"- {p}: **{valores_usuario[p]}** (Límite normativo: {limites_eca[eca_categoria][p]})")
            else:
                st.write("Todos los parámetros se encuentran dentro de los límites.")

    st.markdown("---")
    
    # --- 4. MODELO IA Y TRATAMIENTO SUGERIDO ---
    st.header("🤖 Predicción de Inteligencia Artificial")
    if resultado_ia == "Potable":
        st.success(f"El modelo predictivo clasifica esta muestra como: **{resultado_ia}**")
    elif resultado_ia == "No Potable":
        st.warning(f"El modelo predictivo clasifica esta muestra como: **{resultado_ia}**")
    else:
        st.info(resultado_ia)

    st.header("🛠️ Sugerencia de Tratamiento Requerido")
    # Lógica de tratamiento basada en la categoría que debe alcanzar
    if len(no_cumplen_minsa) == 0:
        st.info("No requiere tratamiento adicional intensivo, el agua ya cumple estándares MINSA (solo asegurar cloración residual).")
    elif len(no_cumplen_eca) == 0:
        if eca_categoria == "A1":
            st.info("El agua de fuente es A1. Sugerencia: **Desinfección (Cloración)** para hacerla potable.")
        elif eca_categoria == "A2":
            st.info("El agua de fuente es A2. Sugerencia: **Tratamiento Convencional** (Coagulación, floculación, decantación, filtración y desinfección).")
        elif eca_categoria == "A3":
            st.info("El agua de fuente es A3. Sugerencia: **Tratamiento Avanzado** (Procesos anteriores + ósmosis inversa, carbón activado u otros procesos especiales).")
    else:
         st.error(f"El agua cruda excede incluso los límites de la categoría ECA {eca_categoria}. Se requiere una revisión del cuerpo receptor o implementar tratamientos de remoción específicos para los parámetros infractores mostrados arriba.")

else:
    st.info("👈 Ingrese los valores en el panel lateral y haga clic en 'Evaluar Calidad de Agua'.")
