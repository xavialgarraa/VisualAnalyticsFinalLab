import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from pandas.api.types import is_numeric_dtype

# --- CONFIG ---
st.set_page_config(
    page_title="Policy Impact Simulator",
    page_icon="🏛️",
    layout="wide",
)

# -------------------------
# LOAD MODEL & METADATA (Reutilizando lógica del Predictor)
# -------------------------
try:
    # Asume que el modelo y los datos están en el mismo directorio
    with open('dropout_model.pkl', 'rb') as file:
        data = pickle.load(file)
    model_loaded = data["model"]
except FileNotFoundError:
    st.error("Error: 'dropout_model.pkl' not found. Please ensure the trained model file is in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# FEATURE COLS (30 columns)
feature_cols = [
    'School', 'Gender', 'Age', 'Address', 'Family_Size', 'Parental_Status', 
    'Mother_Education', 'Father_Education', 'Mother_Job', 'Father_Job', 
    'Reason_for_Choosing_School', 'Guardian', 'Travel_Time', 'Study_Time', 
    'Number_of_Failures', 'School_Support', 'Family_Support', 
    'Extra_Paid_Class', 'Extra_Curricular_Activities', 'Attended_Nursery', 
    'Wants_Higher_Education', 'Internet_Access', 'In_Relationship', 
    'Family_Relationship', 'Free_Time', 'Going_Out', 
    'Weekend_Alcohol_Consumption', 'Weekday_Alcohol_Consumption', 
    'Health_Status', 'Number_of_Absences'
]

# METADATOS (para inputs y descripciones)
feature_info = {
    'School_Support': {'label': "School support", 'help': "Extra educational support from the school."},
    'Family_Support': {'label': "Family educational support", 'help': "Family gives educational support for the student."},
    'Extra_Paid_Class': {'label': "Extra paid classes", 'help': "Participation in extra paid classes."},
    'Study_Time': {'label': "Weekly study time (1-4)", 'help': "Weekly study time (1–4)."},
    'Internet_Access': {'label': "Internet access", 'help': "Availability of internet at home."},
    # Otros features importantes para el perfil
    'Age': {'label': "Age", 'help': "Age of the student."},
    'Number_of_Absences': {'label': "Number of absences", 'help': "Total absences from school."},
    'Number_of_Failures': {'label': "Number of failures", 'help': "Number of past class failures."},
    'Gender': {'label': "Gender", 'help': "M for Male and F for Female."},
}


# LOAD DATASET and Mappings
try:
    df_raw = pd.read_csv("student_dropout.csv").dropna()
except FileNotFoundError:
    st.error("Error: 'student_dropout.csv' not found. Please place the dataset file in the directory.")
    st.stop()

df_num = df_raw.copy()
cat_mappings = {}
typical_values = {}

for col in feature_cols:
    if not is_numeric_dtype(df_raw[col]):
        cat = pd.Categorical(df_raw[col])
        df_num[col] = cat.codes
        cat_mappings[col] = list(cat.categories)
    else:
        df_num[col] = pd.to_numeric(df_raw[col], errors="coerce")
    typical_values[col] = df_num[col].median()

# --- ESTADO DE SESIÓN ---
if 'profile_mode' not in st.session_state:
    st.session_state['profile_mode'] = 'Manual'
if 'base_answers' not in st.session_state:
    st.session_state['base_answers'] = {col: df_raw[col].mode()[0] if col in cat_mappings else typical_values[col] for col in feature_cols}
if 'base_risk' not in st.session_state:
    st.session_state['base_risk'] = None

# -------------------------
# HELPERS
# -------------------------
def input_for_policy_feature(col_name: str, key: str, current_value=None):
    """Generates a Streamlit input widget based on feature type, pre-setting the value."""
    info = feature_info.get(col_name, {'label': col_name, 'help': None})
    label = info['label']
    help_text = info['help']
    
    default_val = current_value
    if default_val is None:
        default_val = df_raw[col_name].mode()[0] if col_name in cat_mappings else typical_values[col_name]


    if col_name in cat_mappings:
        options = cat_mappings[col_name]
        try:
            default_idx = options.index(default_val)
        except:
            default_idx = 0
        return st.selectbox(label, options, index=default_idx, key=key, help=help_text, disabled=(st.session_state['profile_mode'] != 'Manual'))
    else:
        col_min = int(df_num[col_name].min())
        col_max = int(df_num[col_name].max())
        default = int(default_val)
        return st.number_input(
            label,
            min_value=col_min,
            max_value=col_max,
            value=default,
            step=1,
            key=key,
            help=help_text,
            disabled=(st.session_state['profile_mode'] != 'Manual')
        )

def encode_and_predict(answers: dict) -> float:
    """Encodes the feature dictionary into a DataFrame and returns the prediction."""
    sample_list = []
    for col in feature_cols:
        val = answers[col]
        if col in cat_mappings:
            categories = cat_mappings[col]
            if val not in categories:
                code = 0 
            else:
                code = categories.index(val)
            sample_list.append(float(code))
        else:
            # Handle possible float values from inputs
            try:
                sample_list.append(float(val))
            except ValueError:
                # Should not happen with st.number_input, but safety first
                sample_list.append(typical_values[col])
    
    X_sample = pd.DataFrame([sample_list], columns=feature_cols)
    try:
        # Predict probability of dropout (class 1)
        prob = model_loaded.predict_proba(X_sample)[0][1]
    except AttributeError:
        # Fallback if the model does not have predict_proba
        prob = model_loaded.predict(X_sample)[0] 
    return float(prob)

def load_random_student():
    """Carga un estudiante aleatorio del dataset."""
    random_student = df_raw.sample(n=1).iloc[0].to_dict()
    st.session_state['base_answers'] = {col: random_student.get(col, None) for col in feature_cols}
    st.session_state['profile_mode'] = 'Random'
    st.session_state['base_risk'] = None
    st.toast("Estudiante aleatorio cargado.", icon="✅")

def reset_manual_student():
    """Vuelve al perfil manual y lo resetea a valores típicos."""
    st.session_state['base_answers'] = {col: df_raw[col].mode()[0] if col in cat_mappings else typical_values[col] for col in feature_cols}
    st.session_state['profile_mode'] = 'Manual'
    st.session_state['base_risk'] = None
    st.toast("Perfil manual reseteado a valores típicos.", icon="🔄")


# -------------------------
# UI - GOVERNMENT PAGE
# -------------------------

st.title("🏛️ Student Intervention Policy Impact Simulator")
st.write(
    """Esta herramienta gubernamental simula el riesgo de deserción estudiantil y evalúa el impacto potencial 
    de **programas de intervención y becas** en la reducción de ese riesgo. Establezca el perfil del estudiante 
    y luego aplique las políticas para observar el cambio en la probabilidad de abandono.

    **REGLA:** Un riesgo de deserción superior al 50% (0.5) se considera **ALTO** y requiere acción prioritaria.
    """
)
st.divider()

# --- 1. CONFIGURACIÓN DEL PERFIL BASE ---
st.header("1. Definir o Cargar Perfil de Estudiante")

col_op1, col_op2, col_op3 = st.columns([1, 1, 1])

with col_op1:
    if st.button("Cargar Estudiante Aleatorio 🎲", use_container_width=True):
        load_random_student()

with col_op2:
    if st.button("Crear/Editar Perfil Manual 📝", use_container_width=True):
        st.session_state['profile_mode'] = 'Manual'
        st.session_state['base_risk'] = None
        st.toast("Modo Manual Activado.", icon="🖊️")

with col_op3:
    if st.session_state['profile_mode'] == 'Manual' and st.button("Restablecer a Típico 🗑️", use_container_width=True):
        reset_manual_student()


st.info(f"Modo actual: **{st.session_state['profile_mode']}**. Los campos del formulario a continuación están **{'EDITABLES' if st.session_state['profile_mode'] == 'Manual' else 'BLOQUEADOS'}**.")


# Usamos un diccionario temporal para capturar los inputs manuales
current_inputs = st.session_state['base_answers'].copy()

# --- FORMULARIO DE PERFIL ---
st.subheader("Configuración de Variables Clave")

col_s1, col_s2, col_s3 = st.columns(3)

# La función input_for_policy_feature utiliza 'current_inputs[feature]' como valor predefinido.
with col_s1:
    st.markdown("**Demográficos y Rendimiento**")
    current_inputs['Gender'] = input_for_policy_feature('Gender', 'pol_gender', current_inputs['Gender'])
    current_inputs['Age'] = input_for_policy_feature('Age', 'pol_age', current_inputs['Age'])
    current_inputs['Number_of_Absences'] = input_for_policy_feature('Number_of_Absences', 'pol_abs', current_inputs['Number_of_Absences'])
    
with col_s2:
    st.markdown("**Apoyo y Estudio**")
    current_inputs['Family_Support'] = input_for_policy_feature('Family_Support', 'pol_fam_supp', current_inputs['Family_Support'])
    current_inputs['Extra_Paid_Class'] = input_for_policy_feature('Extra_Paid_Class', 'pol_paid_class', current_inputs['Extra_Paid_Class'])
    current_inputs['Internet_Access'] = input_for_policy_feature('Internet_Access', 'pol_internet', current_inputs['Internet_Access'])
    current_inputs['Study_Time'] = input_for_policy_feature('Study_Time', 'pol_study_time', current_inputs['Study_Time'])
    
with col_s3:
    st.markdown("**Factores de Riesgo/Objetivos**")
    current_inputs['Number_of_Failures'] = input_for_policy_feature('Number_of_Failures', 'pol_fail', current_inputs['Number_of_Failures'])
    current_inputs['School_Support'] = input_for_policy_feature('School_Support', 'pol_school_supp', current_inputs['School_Support'])
    current_inputs['Wants_Higher_Education'] = input_for_policy_feature('Wants_Higher_Education', 'pol_higher_ed', current_inputs['Wants_Higher_Education'])

# Actualizar el estado de sesión con los inputs
st.session_state['base_answers'] = current_inputs

# --- Mostrar el perfil cargado/manual completo ---
with st.expander("Ver Perfil Completo del Estudiante (30 Características)"):
    profile_df = pd.DataFrame([st.session_state['base_answers']]).transpose()
    profile_df.columns = ["Valor"]
    st.dataframe(profile_df)


# --- 2. PREDICCIÓN BASE Y VISUALIZACIÓN ---
st.divider()
st.header("2. Evaluación de Riesgo Base")

# Siempre recalcular el riesgo
base_risk = encode_and_predict(st.session_state['base_answers'])
st.session_state['base_risk'] = base_risk
base_pct = base_risk * 100

col_r1, col_r2 = st.columns([1, 2])

with col_r1:
    if base_risk > 0.5:
        box = st.error
        level = "ALTO (¡ACCIÓN REQUERIDA!)"
        emoji = "🔴"
        color = "#e34444"
    elif base_risk > 0.3:
        box = st.warning
        level = "MEDIO"
        emoji = "🟠"
        color = "#ff9a00"
    else:
        box = st.success
        level = "BAJO"
        emoji = "🟢"
        color = "#4CAF50"
    
    box(f"""
        Riesgo de Deserción Base: **{base_pct:.1f}%** Nivel: **{level}** {emoji}
        """)

# --- EXPLICACIÓN DE LA REGLA ---
with col_r2:
    st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 5px solid {color};'>
        <h4>Interpretación del Riesgo:</h4>
        <p>Un riesgo base de **{base_pct:.1f}%** significa que el modelo predice que este estudiante tiene esa probabilidad de abandonar sus estudios.
        <ul>
            <li>Riesgo **Alto (>50%)**: El estudiante tiene más probabilidades de desertar que de continuar. **Intervención inmediata recomendada.**</li>
            <li>Riesgo **Medio (30-50%)**: El estudiante está en una zona de vigilancia; las intervenciones pueden ser preventivas.</li>
            <li>Riesgo **Bajo (<30%)**: Baja probabilidad de deserción.</li>
        </ul>
        </p>
        </div>
    """, unsafe_allow_html=True)


# --- 3. SIMULACIÓN DE INTERVENCIÓN ---
st.header("3. Simulación de Intervención de Políticas")
st.markdown("Aplique las siguientes políticas de becas o apoyo para evaluar su efectividad. La simulación **respeta los valores que el estudiante ya cumple**.")


interventions = {
    "Beca de Apoyo Familiar (F\_Support)": {
        'desc': "Simula un apoyo económico/subsidio que permite a la familia proporcionar soporte educativo (Objetivo: 'Family_Support' = 'yes').",
        'changes': {'Family_Support': 'yes'}
    },
    "Beca de Clases de Refuerzo (Paid\_Class)": {
        'desc': "Financiación directa para clases de refuerzo o tutorías privadas (Objetivo: 'Extra_Paid_Class' = 'yes').",
        'changes': {'Extra_Paid_Class': 'yes'}
    },
    "Programa de Mentoría Escolar (S\_Support)": {
        'desc': "Asignación de un mentor que aumenta el apoyo educativo brindado por la escuela (Objetivo: 'School_Support' = 'yes').",
        'changes': {'School_Support': 'yes'}
    },
    "Beca de Conectividad (Internet)": {
        'desc': "Suministro de acceso a internet de alta velocidad en el hogar (Objetivo: 'Internet_Access' = 'yes').",
        'changes': {'Internet_Access': 'yes'}
    },
    "Programa de Fomento al Estudio (Study\_Time)": {
        'desc': "Incentivo que promueve el estudio autónomo, incrementando el tiempo de estudio semanal al máximo (Objetivo: 'Study_Time' = 4).",
        'changes': {'Study_Time': 4}
    },
    "Intervención Completa (Beca Integral)": {
        'desc': "Combina el apoyo familiar, clases extra, conectividad y tiempo de estudio mejorado (Objetivo: las 4 variables al máximo).",
        'changes': {
            'Family_Support': 'yes', 
            'Extra_Paid_Class': 'yes',
            'Internet_Access': 'yes',
            'Study_Time': 4
        }
    }
}

intervention_cols = st.columns(len(interventions))
simulation_results = []

for i, (name, policy) in enumerate(interventions.items()):
    simulated_answers = st.session_state['base_answers'].copy()
    
    applied_changes = False
    already_met_list = []
    
    # 1. Comprobar y aplicar cambios de la política
    for feature, target_val in policy['changes'].items():
        current_val = st.session_state['base_answers'][feature]
        
        # Check si el valor es numérico o categórico para la comparación
        if feature in cat_mappings:
             is_met = (current_val == target_val)
        else:
            # Para numéricos, asumimos que el valor más alto es el 'óptimo' de la política
            is_met = (current_val == target_val)
        
        if is_met:
            already_met_list.append(feature_info.get(feature, {}).get('label', feature))
            # No se modifica la variable en simulated_answers, ya tiene el valor óptimo.
        else:
            simulated_answers[feature] = target_val
            applied_changes = True

    with intervention_cols[i]:
        st.subheader(f"✅ {name}")
        st.caption(policy['desc'])
        
        # 2. Predecir y calcular métricas
        if not applied_changes:
            # Si no hay cambios aplicados (porque ya cumplía todo)
            simulated_risk = base_risk
            delta = 0
            
            # Mostrar mensaje de estado
            st.markdown(f"**Estado:** No se requiere acción. **Ya cumple la condición** de: *{', '.join(already_met_list)}*.")
        else:
            # Si se aplicó al menos un cambio
            simulated_risk = encode_and_predict(simulated_answers)
            delta = simulated_risk - base_risk
            
            status_note = ""
            if len(already_met_list) > 0:
                status_note = f" (Nota: Ya cumplía: {', '.join(already_met_list)})."
            
            st.markdown(f"**Estado:** Política aplicada.{status_note}")

        
        # 3. Almacenar resultados
        delta_pct = delta * 100
        simulated_pct = simulated_risk * 100
        
        simulation_results.append({
            'Policy': name,
            'Risk_Pct': simulated_pct,
            'Delta_Pct': delta_pct,
            'Description': policy['desc']
        })
        
        # 4. Mostrar Métrica
        if abs(delta) < 0.001:
            st.metric("Nuevo Riesgo", f"{simulated_pct:.1f}%")
            st.caption("Sin cambio significativo")
        else:
            # delta_color="inverse" hace que el rojo sea una subida de riesgo y el verde una bajada.
            st.metric(
                "Nuevo Riesgo",
                f"{simulated_pct:.1f}%",
                delta=f"{delta_pct:+.1f} %",
                delta_color="inverse" 
            )

st.divider()
st.header("4. Análisis de Impacto Agregado")

# --- Visualización de los resultados en un gráfico ---
results_df = pd.DataFrame(simulation_results)
results_df['Risk_Reduction'] = results_df['Delta_Pct'] * -1

# Gráfico de barras que muestra la reducción del riesgo
chart = (
    alt.Chart(results_df)
    .mark_bar()
    .encode(
        y=alt.Y("Policy:N", sort=alt.EncodingSortField(field="Risk_Reduction", order="descending"), title="Política Aplicada"),
        x=alt.X("Risk_Reduction:Q", title="Reducción de Riesgo de Deserción (%)"),
        color=alt.condition(
            alt.datum.Risk_Reduction < 0,
            alt.value("red"),  # Si sube el riesgo
            alt.value("green") # Si baja el riesgo
        ),
        tooltip=["Policy", alt.Tooltip("Risk_Reduction", title="Reducción (%)", format=".1f"), alt.Tooltip("Risk_Pct", title="Riesgo Final (%)", format=".1f")]
    )
    .properties(title=f"Impacto de Políticas vs. Riesgo Base ({base_pct:.1f}%)")
)
st.altair_chart(chart, use_container_width=True)

st.info("""
**Interpretación:** La longitud de la barra indica la reducción de riesgo lograda. 
Las barras verdes a la derecha representan una reducción efectiva, haciendo la política 
una buena candidata para la inversión.
""")

# Pregunta interactiva para el usuario.
st.markdown("---")
st.subheader("Próximo Paso para la Agencia")
st.write("Dado el impacto simulado, ¿qué política de apoyo o beca priorizaría para este tipo de estudiantes?")

if st.button("Explorar el perfil más a fondo en el Predictor 🕵️"):
     st.toast("Redireccionando...", icon="➡️")
     st.markdown(f"Por favor, vaya a la página 'Student Dropout Risk Predictor' para un análisis what-if detallado de las **30 características**.")