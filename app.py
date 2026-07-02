import streamlit as st







import pandas as pd







import numpy as np







import joblib







import os







import gdown















# =========================================================================







# 1. CONFIGURACIÓN DEL MODELO Y GOOGLE DRIVE (¡PON TU ID AQUÍ!)







# =========================================================================















# 🛑 INSTRUCCIÓN: Solo debes reemplazar el texto entre las comillas de abajo 







# por el ID de tu archivo de Google Drive.







# Ejemplo: si tu enlace es https://drive.google.com/file/d/1ABC123xyz.../view







# tu ID es: 1ABC123xyz...







ID_GOOGLE_DRIVE = "1IVhxte4JME6DyFVZ7Dgq9OLMGcc5B21s"















MODEL_PATH = "modelo_agua.pkl"















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







# 2. FUNCIÓN ANTI-BLOQUEO DE DESCARGA (USANDO GDOWN)







# =========================================================================







@st.cache_resource







def load_water_model():







    # Si el archivo no existe en el servidor, lo descarga







    if not os.path.exists(MODEL_PATH):







        if ID_GOOGLE_DRIVE == "PEGA_AQUI_TU_ID_DE_DRIVE":







            st.error("⚠️ ALERTA: No has puesto tu ID de Google Drive en el código. El motor de IA está desactivado.")







            return None







            







        with st.spinner("📥 Descargando el modelo desde Google Drive de forma segura (Solo ocurre la primera vez)..."):







            try:







                # Gdown se salta el bloqueo de virus automáticamente







                url = f'https://drive.google.com/uc?id={ID_GOOGLE_DRIVE}'







                gdown.download(url, MODEL_PATH, quiet=False)







                st.success("✅ Modelo descargado con éxito.")







            except Exception as e:







                st.error(f"Error en la descarga: {e}")







                return None







                







    # Cargar el modelo en memoria







    try:







        return joblib.load(MODEL_PATH)







    except Exception as e:







        st.error(f"Error al leer el archivo .pkl (podría estar corrupto): {e}")







        return None















modelo_pipeline = load_water_model()















# =========================================================================







# 3. DICCIONARIOS NORMATIVOS (MINSA Y ECA 1)







# =========================================================================







NORMATIVA_BASE = {







    'pH': {'minsa': (6.5, 8.5), 'eca_a1': (6.5, 8.5), 'eca_a2': (5.5, 9.0), 'eca_a3': (5.5, 9.0)},







    'CE': {'minsa': 1500.0, 'eca_a1': 1500.0, 'eca_a2': 1600.0, 'eca_a3': 2500.0},







    'T': {'minsa': 35.0, 'eca_a1': 25.0, 'eca_a2': 25.0, 'eca_a3': 25.0}, # T límite referencial







    'OD': {'minsa': 4.0, 'eca_a1': 6.0, 'eca_a2': 5.0, 'eca_a3': 4.0, 'invertido': True}, # > es mejor







    'DBO': {'minsa': 3.0, 'eca_a1': 3.0, 'eca_a2': 5.0, 'eca_a3': 10.0},







    'CT': {'minsa': 0.0, 'eca_a1': 0.0, 'eca_a2': 2000.0, 'eca_a3': 20000.0},







    'AyG': {'minsa': 0.5, 'eca_a1': 0.5, 'eca_a2': 1.0, 'eca_a3': 1.0},







    'ArT': {'minsa': 0.01, 'eca_a1': 0.01, 'eca_a2': 0.01, 'eca_a3': 0.1},







    'PbT': {'minsa': 0.01, 'eca_a1': 0.01, 'eca_a2': 0.01, 'eca_a3': 0.05},







    'CuT': {'minsa': 2.0, 'eca_a1': 2.0, 'eca_a2': 2.0, 'eca_a3': 2.0},







    'MnT': {'minsa': 0.4, 'eca_a1': 0.4, 'eca_a2': 0.4, 'eca_a3': 0.5},







    'Ca': {'minsa': 200.0, 'eca_a1': 200.0, 'eca_a2': 200.0, 'eca_a3': 300.0},







    'Mg': {'minsa': 150.0, 'eca_a1': 150.0, 'eca_a2': 150.0, 'eca_a3': 200.0},







    'Dureza': {'minsa': 500.0, 'eca_a1': 500.0, 'eca_a2': 500.0, 'eca_a3': 500.0}







}















PARAMETROS_ADICIONALES_POOL = {







    'Cadmio (Total) (mg/L)': {'minsa': 0.003, 'eca_a1': 0.003, 'eca_a2': 0.005, 'eca_a3': 0.005, 'default': 0.001},







    'Mercurio (Total) (mg/L)': {'minsa': 0.001, 'eca_a1': 0.001, 'eca_a2': 0.001, 'eca_a3': 0.002, 'default': 0.0005},







    'Hierro (Total) (mg/L)': {'minsa': 0.3, 'eca_a1': 0.3, 'eca_a2': 1.0, 'eca_a3': 5.0, 'default': 0.1},







    'Sulfatos (SO4=) (mg/L)': {'minsa': 250.0, 'eca_a1': 250.0, 'eca_a2': 250.0, 'eca_a3': 500.0, 'default': 80.0},







    'Turbidez (UNT)': {'minsa': 5.0, 'eca_a1': 5.0, 'eca_a2': 100.0, 'eca_a3': 100.0, 'default': 2.0}







}















# =========================================================================







# 4. INTERFAZ: PESTAÑAS







# =========================================================================







tab1, tab2, tab3 = st.tabs([







    "📊 14 Parámetros Base (ML)", 







    "➕ Parámetros Extras (Normativa)", 







    "🔬 Diagnóstico y Tratamiento"







])















inputs = {}















with tab1:







    st.subheader("Parámetros Fisicoquímicos Principales")







    col1, col2, col3 = st.columns(3)







    







    with col1:







        inputs['pH'] = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5, step=0.1)







        inputs['CE'] = st.number_input("Conductividad (µS/cm)", min_value=0.0, value=400.0, step=10.0)







        inputs['T'] = st.number_input("Temperatura (°C)", min_value=0.0, value=18.0, step=0.5)







        inputs['OD'] = st.number_input("Oxígeno Disuelto (mg/L)", min_value=0.0, value=7.5, step=0.1)







        inputs['DBO'] = st.number_input("DBO5 (mg/L)", min_value=0.0, value=2.0, step=0.1)















    with col2:







        inputs['CT'] = st.number_input("Coliformes Termotolerantes (NMP/100mL)", min_value=0.0, value=10.0, step=1.0)







        inputs['AyG'] = st.number_input("Aceites y Grasas (mg/L)", min_value=0.0, value=0.1, step=0.1)







        inputs['ArT'] = st.number_input("Arsénico Total (mg/L)", min_value=0.0, format="%.5f", value=0.002)







        inputs['PbT'] = st.number_input("Plomo Total (mg/L)", min_value=0.0, format="%.5f", value=0.001)







        inputs['CuT'] = st.number_input("Cobre Total (mg/L)", min_value=0.0, format="%.5f", value=0.005)















    with col3:







        inputs['MnT'] = st.number_input("Manganeso Total (mg/L)", min_value=0.0, format="%.5f", value=0.02)







        inputs['Ca'] = st.number_input("Calcio (mg/L)", min_value=0.0, value=45.0, step=1.0)







        inputs['Mg'] = st.number_input("Magnesio (mg/L)", min_value=0.0, value=12.0, step=1.0)







        inputs['Dureza'] = st.number_input("Dureza Total (como CaCO3) (mg/L)", min_value=0.0, value=150.0)















with tab2:







    st.subheader("Robustecer evaluación con parámetros ECA / MINSA")







    st.markdown("Estos parámetros **NO** afectan al modelo de IA, pero **SÍ** se evaluarán en el semáforo legal.")







    







    seleccionados = st.multiselect("Seleccionar extras:", list(PARAMETROS_ADICIONALES_POOL.keys()))







    







    inputs_adicionales = {}







    if seleccionados:







        col_adv1, col_adv2 = st.columns(2)







        for idx, param in enumerate(seleccionados):







            target_col = col_adv1 if idx % 2 == 0 else col_adv2







            with target_col:







                inputs_adicionales[param] = st.number_input(







                    f"{param}", value=PARAMETROS_ADICIONALES_POOL[param]['default'], format="%.4f"







                )















# =========================================================================







# 5. MOTOR DE EVALUACIÓN (PESTAÑA 3)







# =========================================================================







with tab3:







    st.subheader("🔍 Resultados e Informe Técnico")







    







    # --- EVALUACIÓN NORMATIVA ---







    fallas_minsa = []







    eval_dict = {**NORMATIVA_BASE, **{k: PARAMETROS_ADICIONALES_POOL[k] for k in inputs_adicionales}}







    valores_totales = {**inputs, **inputs_adicionales}







    







    # 1. Chequeo MINSA







    for param, limits in eval_dict.items():







        val = valores_totales[param]







        minsa_lim = limits['minsa']







        







        if param == 'pH':







            if val < minsa_lim[0] or val > minsa_lim[1]: fallas_minsa.append((param, val, f"Rango: {minsa_lim}"))







        elif limits.get('invertido', False):







            if val < minsa_lim: fallas_minsa.append((param, val, f"Mínimo: {minsa_lim}"))







        else:







            if val > minsa_lim: fallas_minsa.append((param, val, f"Máximo: {minsa_lim}"))















    es_potable_minsa = len(fallas_minsa) == 0















    # 2. Chequeo ECA (Si falla MINSA)







    peor_categoria_eca = "A1"







    detalles_eca = []















    if not es_potable_minsa:







        for param, limits in eval_dict.items():







            val = valores_totales[param]







            if param == 'pH':







                if limits['eca_a1'][0] <= val <= limits['eca_a1'][1]: cat = "A1"







                elif limits['eca_a2'][0] <= val <= limits['eca_a2'][1]: cat = "A2"







                else: cat = "A3"







            elif limits.get('invertido', False):







                if val >= limits['eca_a1']: cat = "A1"







                elif val >= limits['eca_a2']: cat = "A2"







                elif val >= limits['eca_a3']: cat = "A3"







                else: cat = "EXCEDE A3"







            else:







                if val <= limits['eca_a1']: cat = "A1"







                elif val <= limits['eca_a2']: cat = "A2"







                elif val <= limits['eca_a3']: cat = "A3"







                else: cat = "EXCEDE A3"







            







            detalles_eca.append({'Parámetro': param, 'Valor': val, 'Categoría': cat})







            order = {"A1": 1, "A2": 2, "A3": 3, "EXCEDE A3": 4}







            if order[cat] > order[peor_categoria_eca]:







                peor_categoria_eca = cat















    # --- PREDICCIÓN MACHINE LEARNING ---







    ml_result_text = "Desactivado (Revisa ID de Drive)"







    prob_potable = 0.0







    







    if modelo_pipeline is not None:







        features_order = ['pH', 'CE', 'T', 'OD', 'DBO', 'CT', 'AyG', 'ArT', 'PbT', 'CuT', 'MnT', 'Ca', 'Mg', 'Dureza']







        features_vector = pd.DataFrame([[inputs[feat] for feat in features_order]], columns=features_order)







        







        try:







            # Según tu cuaderno, utilizamos UMBRAL_SEGURIDAD = 0.75







            prob = modelo_pipeline.predict_proba(features_vector)[0]







            prob_potable = prob[1]







            pred = 1 if prob_potable >= 0.75 else 0







            







            if pred == 1:







                ml_result_text = f"Potable (Prob: {prob_potable*100:.1f}%)"







            else:







                ml_result_text = f"No Potable (Prob: {prob[0]*100:.1f}%)"







        except Exception as e:







            ml_result_text = f"Error IA: {e}"















    # --- RENDERIZADO VISUAL ---







    col_res1, col_res2 = st.columns(2)







    







    with col_res1:







        st.markdown("#### 🏛️ Evaluación Legal (MINSA)")







        if es_potable_minsa:







            st.markdown("""<div class="card-potable"><h4>✅ CUMPLE MINSA</h4><p>Apto para consumo directo.</p></div>""", unsafe_allow_html=True)







        else:







            st.markdown(f"""<div class="card-nopotable"><h4>❌ NO CUMPLE MINSA</h4><p>Violación en {len(fallas_minsa)} parámetros.</p></div>""", unsafe_allow_html=True)







            with st.expander("Ver variables infractoras"):







                st.dataframe(pd.DataFrame(fallas_minsa, columns=['Variable', 'Valor', 'Límite']))















    with col_res2:







        st.markdown("#### 🤖 Predicción IA (Random Forest / GB)")







        if modelo_pipeline is not None:







            st.metric(label="Clasificador (Umbral Seguridad 0.75)", value=ml_result_text)







            st.progress(float(prob_potable), text="Probabilidad general de Potabilidad")















    # --- RECOMENDACIÓN ECA ---







    if not es_potable_minsa:







        st.markdown("---")







        st.markdown(f"### 🛠️ Plan de Tratamiento (ECA Categoría 1 - Peor Nivel: {peor_categoria_eca})")







        







        if peor_categoria_eca == "A1":







            st.markdown("""<div class="treatment-box">🟢 <b>Subcategoría A1: Desinfección Simple.</b><br>Requiere cloración estándar u ozonización para eliminar carga microbiológica antes de distribuir.</div>""", unsafe_allow_html=True)







        elif peor_categoria_eca == "A2":







            st.markdown("""<div class="treatment-box" style="border-left-color: #FD7E14; background-color: #FFE8D6;">🟠 <b>Subcategoría A2: Tratamiento Convencional.</b><br>Implementar Coagulación, Floculación, Sedimentación, Filtración rápida y Desinfección.</div>""", unsafe_allow_html=True)







        elif peor_categoria_eca == "A3":







            st.markdown("""<div class="treatment-box" style="border-left-color: #DC3545; background-color: #F8D7DA;">🔴 <b>Subcategoría A3: Tratamiento Avanzado.</b><br>Contaminación alta. Requiere Ósmosis Inversa, Carbón Activado o Procesos de Oxidación Avanzada.</div>""", unsafe_allow_html=True)







        else:







            st.markdown("""<div class="treatment-box" style="background-color: #343A40; color: #FFF; border-left-color: #000;">⚫ <b>EXCEDE A3: ALERTA CRÍTICA.</b><br>El agua supera la capacidad de las plantas potabilizadoras convencionales. Descarte inmediato para consumo.</div>""", unsafe_allow_html=True)















        with st.expander("Ver desglose ECA completo"):







            st.dataframe(pd.DataFrame(detalles_eca))

