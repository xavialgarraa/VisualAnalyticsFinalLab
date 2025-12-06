import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------
# LOAD MODEL + ENCODERS
# -------------------------
with open('dropout_model.pkl', 'rb') as file:
    data = pickle.load(file)

model_loaded = data["model"]
encoders = data["encoders"]
feature_cols = data["feature_cols"]

# -------------------------
# LOAD DATASET TO GET TYPICAL VALUES (MEDIANS / MODES)
# -------------------------
df = pd.read_csv("student_management_dataset.csv")


with st.expander("Show dataset"):
    st.dataframe(df.head(200))
# Valors típics per a TOTES les columnes (no només algunes)
typical_values = {}
for col in feature_cols:
    if col in encoders:
        # Categòrica: usem el valor més freqüent (mode)
        typical_values[col] = df[col].mode()[0]
    else:
        # Numèrica: usem la mediana
        typical_values[col] = df[col].median()

# Opcions de les variables categòriques des dels encoders (si existeixen)
gender_options = list(encoders['Gender'].classes_)
ethnicity_options = list(encoders['Ethnicity'].classes_)
socio_options = list(encoders['Socioeconomic_Status'].classes_)
parentedu_options = list(encoders['Parental_Education_Level'].classes_)

# Per altres categòriques que puguin existir (si n'hi ha al diccionari)
other_cat_options = {}
for col in encoders:
    if col not in ['Gender', 'Ethnicity', 'Socioeconomic_Status', 'Parental_Education_Level']:
        other_cat_options[col] = list(encoders[col].classes_)

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="wide")

st.title("Student Dropout Risk Predictor 🎓")
st.write(
    "You can edit any of the 21 features used by the model. "
    "If you leave the default values, the app will use typical (median/mode) values from the dataset."
)

# Min / max per a algunes numèriques (la resta les fem servir directament amb min/max del df)
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
hours_min, hours_max = int(df['Study_Hours_per_Week'].min()), int(df['Study_Hours_per_Week'].max())
extra_min, extra_max = int(df['Extracurricular_Activities'].min()), int(df['Extracurricular_Activities'].max())

# Attendance el limitem a [0, 100] per seguretat
att_min, att_max = 0, 100
att_med = typical_values['Attendance_Rate']
if pd.isna(att_med):
    att_med = 75
att_med = max(0, min(100, int(att_med)))

col1, col2, col3, col4 = st.columns(4)

# ------------- COL1: Perfil bàsic -------------
with col1:
    # Gender
    gender_default = gender_options.index(typical_values['Gender']) if typical_values['Gender'] in gender_options else 0
    gender = st.selectbox("Gender", gender_options, index=gender_default, key="pp_gender")

    # Age
    age = st.slider(
        "Age",
        min_value=age_min,
        max_value=age_max,
        value=int(typical_values['Age']),
        key="pp_age"
    )

    # Ethnicity
    eth_default = ethnicity_options.index(typical_values['Ethnicity']) if typical_values['Ethnicity'] in ethnicity_options else 0
    ethnicity = st.selectbox("Ethnicity", ethnicity_options, index=eth_default, key="pp_ethnicity")

    # Socioeconomic Status
    socio_default = socio_options.index(typical_values['Socioeconomic_Status']) if typical_values['Socioeconomic_Status'] in socio_options else 0
    socio = st.selectbox("Socioeconomic Status", socio_options, index=socio_default, key="pp_socio")

    # Parental Education Level
    pedu_default = parentedu_options.index(typical_values['Parental_Education_Level']) if typical_values['Parental_Education_Level'] in parentedu_options else 0
    parentedu = st.selectbox("Parental Education Level", parentedu_options, index=pedu_default, key="pp_parentedu")

# ------------- COL2: Estat acadèmic -------------
with col2:
    # Disability_Status (numèrica o categòrica)
    if 'Disability_Status' in encoders:
        options = other_cat_options['Disability_Status']
        dis_default = options.index(typical_values['Disability_Status']) if typical_values['Disability_Status'] in options else 0
        disability = st.selectbox("Disability Status", options, index=dis_default, key="pp_disability")
    else:
        disability = st.number_input(
            "Disability Status",
            value=float(typical_values['Disability_Status']),
            step=1.0,
            key="pp_disability"
        )

    # GPA
    gpa = st.number_input(
        "GPA",
        value=float(typical_values['GPA']),
        step=0.1,
        key="pp_gpa"
    )

    # Past Academic Performance
    if 'Past_Academic_Performance' in encoders:
        options = other_cat_options.get('Past_Academic_Performance', list(df['Past_Academic_Performance'].unique()))
        pap_default = options.index(typical_values['Past_Academic_Performance']) if typical_values['Past_Academic_Performance'] in options else 0
        past_perf = st.selectbox("Past Academic Performance", options, index=pap_default, key="pp_past_perf")
    else:
        past_perf = st.number_input(
            "Past Academic Performance",
            value=float(typical_values['Past_Academic_Performance']),
            step=1.0,
            key="pp_past_perf"
        )

    # Current Semester Performance
    if 'Current_Semester_Performance' in encoders:
        options = other_cat_options.get('Current_Semester_Performance', list(df['Current_Semester_Performance'].unique()))
        csp_default = options.index(typical_values['Current_Semester_Performance']) if typical_values['Current_Semester_Performance'] in options else 0
        current_perf = st.selectbox("Current Semester Performance", options, index=csp_default, key="pp_current_perf")
    else:
        current_perf = st.number_input(
            "Current Semester Performance",
            value=float(typical_values['Current_Semester_Performance']),
            step=1.0,
            key="pp_current_perf"
        )

    # Courses Failed
    courses_failed = st.number_input(
        "Courses Failed",
        value=float(typical_values['Courses_Failed']),
        step=1.0,
        key="pp_courses_failed"
    )

# ------------- COL3: Hores / assistència / participació -------------
with col3:
    # Credits Completed
    credits_completed = st.number_input(
        "Credits Completed",
        value=float(typical_values['Credits_Completed']),
        step=1.0,
        key="pp_credits"
    )

    # Study Hours
    study_hours = st.slider(
        "Study Hours per Week",
        min_value=hours_min,
        max_value=hours_max,
        value=int(typical_values['Study_Hours_per_Week']),
        key="pp_study_hours"
    )

    # Attendance
    attendance = st.slider(
        "Attendance Rate (%)",
        min_value=att_min,
        max_value=att_max,
        value=att_med,
        key="pp_attendance"
    )

    # Number of Late Submissions
    late_sub = st.number_input(
        "Number of Late Submissions",
        value=float(typical_values['Number_of_Late_Submissions']),
        step=1.0,
        key="pp_late"
    )

    # Class Participation Score
    class_part = st.number_input(
        "Class Participation Score",
        value=float(typical_values['Class_Participation_Score']),
        step=1.0,
        key="pp_class_part"
    )

# ------------- COL4: Altres factors -------------
with col4:
    # Online Learning Hours
    online_hours = st.number_input(
        "Online Learning Hours",
        value=float(typical_values['Online_Learning_Hours']),
        step=1.0,
        key="pp_online"
    )

    # Library Usage Hours
    library_hours = st.number_input(
        "Library Usage Hours",
        value=float(typical_values['Library_Usage_Hours']),
        step=1.0,
        key="pp_library"
    )

    # Disciplinary Actions
    discipl = st.number_input(
        "Disciplinary Actions",
        value=float(typical_values['Disciplinary_Actions']),
        step=1.0,
        key="pp_disciplinary"
    )

    # Social Engagement Score
    social_score = st.number_input(
        "Social Engagement Score",
        value=float(typical_values['Social_Engagement_Score']),
        step=1.0,
        key="pp_social"
    )

    # Mental Health Score
    mental_score = st.number_input(
        "Mental Health Score",
        value=float(typical_values['Mental_Health_Score']),
        step=1.0,
        key="pp_mental"
    )

    # Extracurricular Activities
    extra = st.slider(
        "Extracurricular Activities",
        min_value=extra_min,
        max_value=extra_max,
        value=int(typical_values['Extracurricular_Activities']),
        key="pp_extra"
    )

st.divider()

if st.button("Predict Dropout Risk"):
    try:
        # Guardem totes les respostes de l’usuari (ja són 21 features)
        user_answers = {
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'Socioeconomic_Status': socio,
            'Parental_Education_Level': parentedu,
            'Disability_Status': disability,
            'GPA': gpa,
            'Past_Academic_Performance': past_perf,
            'Current_Semester_Performance': current_perf,
            'Courses_Failed': courses_failed,
            'Credits_Completed': credits_completed,
            'Study_Hours_per_Week': study_hours,
            'Attendance_Rate': attendance,
            'Number_of_Late_Submissions': late_sub,
            'Class_Participation_Score': class_part,
            'Online_Learning_Hours': online_hours,
            'Library_Usage_Hours': library_hours,
            'Disciplinary_Actions': discipl,
            'Social_Engagement_Score': social_score,
            'Mental_Health_Score': mental_score,
            'Extracurricular_Activities': extra
        }

        # Construir el vector de features en l'ordre correcte
        sample_list = []
        for col in feature_cols:
            val = user_answers[col]

            if col in encoders:  # categòrica → encodar
                val_enc = encoders[col].transform([val])[0]
                sample_list.append(val_enc)
            else:  # numèrica
                sample_list.append(float(val))

        X_sample = pd.DataFrame([sample_list], columns=feature_cols)

        # Predicció
        y_pred = model_loaded.predict(X_sample)[0]

        st.subheader("Predicted Dropout Risk")
        st.metric("Dropout risk (0–1)", f"{y_pred:.3f}")

        # Missatge qualitatiu
        if y_pred < 0.3:
            st.success("Risk level: **LOW** 🟢")
        elif y_pred < 0.6:
            st.warning("Risk level: **MEDIUM** 🟠")
        else:
            st.error("Risk level: **HIGH** 🔴")

        # Guardar per explicabilitat si vols usar-ho després
        st.session_state["explain_X_display"] = pd.DataFrame([user_answers])
        st.session_state["explain_X_encoded"] = X_sample
        st.session_state["explain_y_pred"] = float(y_pred)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
