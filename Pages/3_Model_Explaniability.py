import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import pickle
from pandas.api.types import is_numeric_dtype

# -------------------------
# CONFIG STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Student Dropout – Model Explainability",
    page_icon="🎓",
    layout="wide",
)

st.title("Model Explainability 🧑‍🏫")
st.write(
    """
    Welcome to the **model explainability** section for the *Student Dropout Risk Predictor*.

    On this page you can understand **how the model works** and **why** it predicts a 
    certain dropout risk for a given student, using **SHAP (SHapley Additive exPlanations)** values.

    You can explore:
    - **SHAP introduction:** what SHAP values are and what dataset is used.
    - **Global behaviour:** which features matter most overall and how dropout risk reacts to some key variables.
    - **Local explanations:** why the model predicted that specific dropout risk for a student.
    """
)
st.divider()

# -------------------------
# LOAD MODEL
# -------------------------
with open("dropout_model.pkl", "rb") as file:
    data = pickle.load(file)

model_loaded = data["model"]

# -------------------------
# FEATURES (same as 2_Dropout_Predictor)
# -------------------------
feature_cols = [
    "School",
    "Gender",
    "Age",
    "Address",
    "Family_Size",
    "Parental_Status",
    "Mother_Education",
    "Father_Education",
    "Mother_Job",
    "Father_Job",
    "Reason_for_Choosing_School",
    "Guardian",
    "Travel_Time",
    "Study_Time",
    "Number_of_Failures",
    "School_Support",
    "Family_Support",
    "Extra_Paid_Class",
    "Extra_Curricular_Activities",
    "Attended_Nursery",
    "Wants_Higher_Education",
    "Internet_Access",
    "In_Relationship",
    "Family_Relationship",
    "Free_Time",
    "Going_Out",
    "Weekend_Alcohol_Consumption",
    "Weekday_Alcohol_Consumption",
    "Health_Status",
    "Number_of_Absences",
]

# -------------------------
# METADATA (labels + help) – same estil que 2_Dropout_Predictor
# -------------------------
feature_info = {
    'School': {'label': "School", 'help': "Name/code of the school attended."},
    'Gender': {'label': "Gender", 'help': "M for Male and F for Female."},
    'Age': {'label': "Age", 'help': "Age of the student."},
    'Address': {'label': "Address", 'help': "U for urban and R for rural."},
    'Family_Size': {'label': "Family size", 'help': "GT3 for >3 and LE3 for ≤3."},
    'Parental_Status': {'label': "Parental status", 'help': "A for together and T for apart."},
    'Mother_Education': {'label': "Mother education level", 'help': "0–4 (higher means more education)."},
    'Father_Education': {'label': "Father education level", 'help': "0–4 (higher means more education)."},
    'Mother_Job': {'label': "Mother job", 'help': "Type of job held by the mother."},
    'Father_Job': {'label': "Father job", 'help': "Type of job held by the father."},
    'Reason_for_Choosing_School': {'label': "Reason for choosing school", 'help': "E.g., course, reputation, etc."},
    'Guardian': {'label': "Guardian", 'help': "Guardian of the student (e.g., mother, father)."},
    'Travel_Time': {'label': "Travel time (minutes)", 'help': "Minutes taken to travel to school."},
    'Study_Time': {'label': "Weekly study time", 'help': "Weekly study time (1–4)."},
    'Number_of_Failures': {'label': "Number of failures", 'help': "Number of past class failures."},
    'School_Support': {'label': "School support", 'help': "Extra educational support from the school."},
    'Family_Support': {'label': "Family educational support", 'help': "Family gives educational support for the student."},
    'Extra_Paid_Class': {'label': "Extra paid classes", 'help': "Participation in extra paid classes."},
    'Extra_Curricular_Activities': {'label': "Extracurricular activities", 'help': "Involvement in activities."},
    'Attended_Nursery': {'label': "Nursery attendance", 'help': "Attendance in nursery school."},
    'Wants_Higher_Education': {'label': "Higher education intention", 'help': "Desire to pursue higher education."},
    'Internet_Access': {'label': "Internet access", 'help': "Availability of internet at home."},
    'In_Relationship': {'label': "Relationship status", 'help': "Involved in a romantic relationship."},
    'Family_Relationship': {'label': "Family relationship quality", 'help': "Quality (1–5)."},
    'Free_Time': {'label': "Free time", 'help': "Free time after school (1–5)."},
    'Going_Out': {'label': "Going out", 'help': "Going out with friends (1–5)."},
    'Weekend_Alcohol_Consumption': {'label': "Weekend alcohol consumption", 'help': "Scale 1–5."},
    'Weekday_Alcohol_Consumption': {'label': "Weekday alcohol consumption", 'help': "Scale 1–5."},
    'Health_Status': {'label': "Health status", 'help': "Health rating (1–5)."},
    'Number_of_Absences': {'label': "Number of absences", 'help': "Total absences from school."}
}

# -------------------------
# LOAD & ENCODE DATASET (USED AS TEST SET FOR SHAP)
# -------------------------
df_raw = pd.read_csv("student_dropout.csv")

df_num = df_raw.copy()
cat_mappings = {}

for col in feature_cols:
    if not is_numeric_dtype(df_raw[col]):
        cat = pd.Categorical(df_raw[col])
        df_num[col] = cat.codes
        cat_mappings[col] = list(cat.categories)
    else:
        df_num[col] = pd.to_numeric(df_raw[col], errors="coerce")

typical_values = {col: df_num[col].median() for col in feature_cols}

X_test = df_num[feature_cols].copy()

if "model_loaded" not in globals():
    st.error("The model object is not available. Please load it before using this page.")
    st.stop()
if "X_test" not in globals():
    st.error("The dataset `X_test` is not available. Please define or load it before using this page.")
    st.stop()

# -------------------------
# HELPERS (reuse style of 2_Dropout_Predictor)
# -------------------------
def input_for_feature(col_name: str, key: str):
    """
    Generic input control for a feature:
    - Categorical → selectbox with categories from the data.
    - Numeric → slider between min and max observed values.
    """
    info = feature_info[col_name]
    label = info['label']
    help_text = info['help']

    if col_name in cat_mappings:
        options = cat_mappings[col_name]
        try:
            default_label = df_raw[col_name].mode()[0]
            default_idx = options.index(default_label) if default_label in options else 0
        except Exception:
            default_idx = 0
        return st.selectbox(label, options, index=default_idx, key=key, help=help_text)
    else:
        col_min = int(df_num[col_name].min())
        col_max = int(df_num[col_name].max())
        default = int(typical_values[col_name])
        return st.slider(
            label,
            min_value=col_min,
            max_value=col_max,
            value=default,
            step=1,
            key=key,
            help=help_text
        )

def encode_for_model(answers: dict) -> pd.DataFrame:
    """
    Converteix un diccionari de respostes (valors humans) a un DataFrame numèric
    amb el mateix esquema que X_test, llest per al model.
    """
    sample_list = []
    for col in feature_cols:
        val = answers[col]
        if col in cat_mappings:
            categories = cat_mappings[col]
            code = categories.index(val)
            sample_list.append(float(code))
        else:
            sample_list.append(float(val))
    X_sample = pd.DataFrame([sample_list], columns=feature_cols)
    return X_sample

# -------------------------
# SHAP INTRODUCTION
# -------------------------
st.subheader("SHAP Explainer")
st.write(
    """
    **What are SHAP values?**  
    SHAP (SHapley Additive exPlanations) values come from cooperative game theory.
    Each feature is treated like a “player” in a game, and its SHAP value measures
    how much that feature **pushes the prediction up or down** compared to a
    baseline (the model’s average prediction).  

    - Positive SHAP value → the feature makes the predicted **dropout risk higher**.  
    - Negative SHAP value → the feature makes the predicted **dropout risk lower**.  

    The sum of all SHAP values for a student, plus the baseline, equals the final
    predicted dropout risk for that student.
    """
)

with st.spinner("Computing SHAP values for the dataset..."):
    shap.initjs()
    explainer = shap.TreeExplainer(model_loaded)
    shap_values = explainer(X_test)

st.write(
    f"The dataset used here has **{X_test.shape[0]} students** and **{X_test.shape[1]} features**."
)

y_pred_all = model_loaded.predict(X_test)
avg_risk_pred = float(np.mean(y_pred_all))
st.metric("Average predicted dropout risk in the dataset", f"{avg_risk_pred * 100:.1f}%")
st.divider()

# -------------------------
# GLOBAL EXPLAINABILITY
# -------------------------
st.subheader("Global Explainability")
gcol1, gcol2 = st.columns(2)

with gcol1:
    st.write("### SHAP summary plot (feature impact)")
    st.write(
        """
        Each point is a student in the dataset.  
        The color shows the feature value (red = high, blue = low).  
        The horizontal axis shows how much that feature pushed the dropout risk 
        **up or down**.
        """
    )
    fig1, _ = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig1)

with gcol2:
    st.write("### SHAP bar plot (mean absolute importance)")
    st.write(
        """
        This bar chart shows which features are **most important on average**
        for the model’s dropout risk predictions.
        """
    )
    fig2, _ = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig2)

# -------------------------
# FEATURE-WISE SCATTER PLOTS
# -------------------------
st.subheader("Feature-wise SHAP scatter plots")
st.write(
    """
    These plots show how the SHAP value (impact on dropout risk) changes with the 
    feature value for some key variables:
    - **Study_Time**
    - **Number_of_Failures**
    - **Number_of_Absences**
    """
)

fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    if "Study_Time" in X_test.columns:
        st.write("#### SHAP vs Study_Time")
        plt.figure()
        shap.plots.scatter(shap_values[:, "Study_Time"], show=False)
        fig_s = plt.gcf()
        st.pyplot(fig_s)
        plt.close(fig_s)
    else:
        st.info("Column 'Study_Time' not found in X_test.")

with fcol2:
    if "Number_of_Failures" in X_test.columns:
        st.write("#### SHAP vs Number_of_Failures")
        plt.figure()
        shap.plots.scatter(shap_values[:, "Number_of_Failures"], show=False)
        fig_f = plt.gcf()
        st.pyplot(fig_f)
        plt.close(fig_f)
    else:
        st.info("Column 'Number_of_Failures' not found in X_test.")

with fcol3:
    if "Number_of_Absences" in X_test.columns:
        st.write("#### SHAP vs Number_of_Absences")
        plt.figure()
        shap.plots.scatter(shap_values[:, "Number_of_Absences"], show=False)
        fig_a = plt.gcf()
        st.pyplot(fig_a)
        plt.close(fig_a)
    else:
        st.info("Column 'Number_of_Absences' not found in X_test.")

# -------------------------
# DEPENDENCE PLOTS WITH INTERACTIONS
# -------------------------
st.subheader("Dependence plots with interactions")
st.write(
    """
    Dependence plots reveal **non-linear effects** and interactions.  
    Here we inspect how:
    - **Number_of_Failures** interacts with **Study_Time**
    - **Number_of_Absences** interacts with **Study_Time**
    """
)

dcol1, dcol2 = st.columns(2)

with dcol1:
    if {"Number_of_Failures", "Study_Time"}.issubset(X_test.columns):
        st.write("#### Number_of_Failures vs Study_Time interaction")
        plt.figure()
        shap.dependence_plot(
            "Number_of_Failures",
            shap_values.values,
            X_test,
            interaction_index="Study_Time",
            show=False,
        )
        fig_dep1 = plt.gcf()
        st.pyplot(fig_dep1)
        plt.close(fig_dep1)
    else:
        st.info("Columns 'Number_of_Failures' and/or 'Study_Time' not found in X_test.")

with dcol2:
    if {"Number_of_Absences", "Study_Time"}.issubset(X_test.columns):
        st.write("#### Number_of_Absences vs Study_Time interaction")
        plt.figure()
        shap.dependence_plot(
            "Number_of_Absences",
            shap_values.values,
            X_test,
            interaction_index="Study_Time",
            show=False,
        )
        fig_dep2 = plt.gcf()
        st.pyplot(fig_dep2)
        plt.close(fig_dep2)
    else:
        st.info("Columns 'Number_of_Absences' and/or 'Study_Time' not found in X_test.")

st.divider()

# -------------------------
# LOCAL EXPLAINABILITY
# -------------------------
st.subheader("Local Explainability")
st.write(
    """
    Here you can see **why the model predicted a given dropout risk** for a specific student.

    You can:
    - Explain the **last student you predicted** in the Dropout Predictor page.
    - **Pick a student** from the dataset used as test set.
    - Or **enter a new student manually** and explain that custom profile.
    """
)

shap_values_test = shap_values

mode = st.radio(
    "Which student do you want to explain?",
    (
        "Explain your last predicted student (from Dropout Predictor)",
        "Pick a student from the dataset",
        "Enter a new student manually",
    ),
)

X_explain = None
X_explain_display = None
y_explain_pred = None
shap_row = None

has_last_pred = (
    "explain_X_display" in st.session_state
    and "explain_X_encoded" in st.session_state
    and "explain_y_pred" in st.session_state
    and isinstance(st.session_state["explain_X_encoded"], pd.DataFrame)
    and len(st.session_state["explain_X_encoded"]) > 0
)

use_test_set = False
use_manual = False

# --------- OPTION 1: LAST PREDICTED STUDENT (FROM PREDICTOR PAGE) ---------
if mode == "Explain your last predicted student (from Dropout Predictor)":
    if has_last_pred:
        X_explain = st.session_state["explain_X_encoded"]
        X_explain_display = st.session_state["explain_X_display"]
        y_explain_pred = st.session_state["explain_y_pred"]

        shap_values_explain = explainer(X_explain)
        shap_row = shap_values_explain[0]

        st.info(
            "You are explaining the **last student you predicted** in the Dropout Predictor page."
        )
    else:
        st.warning(
            "No previous prediction found from the Dropout Predictor page. "
            "Please predict a student there first, or use one of the other options below."
        )
        use_test_set = True

# --------- OPTION 2: PICK A STUDENT FROM THE DATASET ---------
elif mode == "Pick a student from the dataset":
    use_test_set = True

if use_test_set and mode != "Enter a new student manually":
    instance_index = st.slider(
        "Select a student index from the dataset",
        min_value=0,
        max_value=X_test.shape[0] - 1,
        value=0,
        step=1,
        help="Choose which student from the dataset you want to inspect.",
    )

    X_explain = X_test.iloc[[instance_index]]

    # Build a human-readable version using categorical mappings
    X_explain_display = X_explain.copy()
    for col in cat_mappings:
        try:
            X_explain_display[col] = (
                X_explain_display[col]
                .astype(int)
                .map(
                    lambda idx, cats=cat_mappings[col]: cats[idx]
                    if 0 <= idx < len(cats)
                    else "Unknown"
                )
            )
        except Exception:
            pass

    y_explain_pred = float(model_loaded.predict(X_explain)[0])
    shap_row = shap_values_test[instance_index]

    st.info(f"You are explaining **student #{instance_index} from the dataset**.")

# --------- OPTION 3: ENTER A NEW STUDENT MANUALLY ---------
if mode == "Enter a new student manually":
    use_manual = True
    st.info("Fill in the fields below to define a new student profile, then click **Explain this student**.")

    with st.form("manual_student_form"):
        # PERSONAL
        st.header("👤 Personal information")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p11, col_p22, col_p33, col_p44 = st.columns(4)
        col_p111, col_p222, col_p333, col_p444 = st.columns(4)

        with col_p1:
            gender = input_for_feature("Gender", "man_gender")
            address = input_for_feature("Address", "man_address")

        with col_p2:
            age = input_for_feature("Age", "man_age")

        with col_p11:
            free_time = input_for_feature("Free_Time", "man_free_time")
            in_relationship = input_for_feature("In_Relationship", "man_in_relationship")

        with col_p22:
            going_out = input_for_feature("Going_Out", "man_going_out")

        with col_p111:
            wkday_alc = input_for_feature("Weekday_Alcohol_Consumption", "man_wkday_alc")
            health = input_for_feature("Health_Status", "man_health")

        with col_p222:
            wknd_alc = input_for_feature("Weekend_Alcohol_Consumption", "man_wknd_alc")

        st.markdown("---")

        # FAMILY
        st.header("🏠 Family information")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f11, col_f22, col_f33, col_f44 = st.columns(4)

        with col_f1:
            family_size = input_for_feature("Family_Size", "man_family_size")
            parental_status = input_for_feature("Parental_Status", "man_parental_status")

        with col_f2:
            mother_edu = input_for_feature("Mother_Education", "man_mother_edu")
            father_edu = input_for_feature("Father_Education", "man_father_edu")

        with col_f3:
            mother_job = input_for_feature("Mother_Job", "man_mother_job")
            father_job = input_for_feature("Father_Job", "man_father_job")

        with col_f11:
            guardian = input_for_feature("Guardian", "man_guardian")
            family_rel = input_for_feature("Family_Relationship", "man_family_rel")

        st.markdown("---")

        # ACADEMIC
        st.header("📚 Academic information")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a11, col_a22, col_a33, col_a44 = st.columns(4)

        with col_a1:
            school = input_for_feature("School", "man_school")
            reason_school = input_for_feature("Reason_for_Choosing_School", "man_reason_school")

        with col_a2:
            travel_time = input_for_feature("Travel_Time", "man_travel_time")
            study_time = input_for_feature("Study_Time", "man_study_time")

        with col_a3:
            num_failures = input_for_feature("Number_of_Failures", "man_num_failures")
            nursery = input_for_feature("Attended_Nursery", "man_nursery")
            higher_ed = input_for_feature("Wants_Higher_Education", "man_higher_ed")

        with col_a11:
            school_support = input_for_feature("School_Support", "man_school_support")
            family_support = input_for_feature("Family_Support", "man_family_support")

        with col_a22:
            extra_paid = input_for_feature("Extra_Paid_Class", "man_extra_paid")
            extra_curr = input_for_feature("Extra_Curricular_Activities", "man_extra_curr")

        with col_a33:
            internet = input_for_feature("Internet_Access", "man_internet")
            absences = input_for_feature("Number_of_Absences", "man_absences")

        submitted = st.form_submit_button("Explain this student")

    if submitted:
        current_inputs = {
            'School': school,
            'Gender': gender,
            'Age': age,
            'Address': address,
            'Family_Size': family_size,
            'Parental_Status': parental_status,
            'Mother_Education': mother_edu,
            'Father_Education': father_edu,
            'Mother_Job': mother_job,
            'Father_Job': father_job,
            'Reason_for_Choosing_School': reason_school,
            'Guardian': guardian,
            'Travel_Time': travel_time,
            'Study_Time': study_time,
            'Number_of_Failures': num_failures,
            'School_Support': school_support,
            'Family_Support': family_support,
            'Extra_Paid_Class': extra_paid,
            'Extra_Curricular_Activities': extra_curr,
            'Attended_Nursery': nursery,
            'Wants_Higher_Education': higher_ed,
            'Internet_Access': internet,
            'In_Relationship': in_relationship,
            'Family_Relationship': family_rel,
            'Free_Time': free_time,
            'Going_Out': going_out,
            'Weekend_Alcohol_Consumption': wknd_alc,
            'Weekday_Alcohol_Consumption': wkday_alc,
            'Health_Status': health,
            'Number_of_Absences': absences,
        }

        X_explain = encode_for_model(current_inputs)
        X_explain_display = pd.DataFrame([current_inputs])
        y_explain_pred = float(model_loaded.predict(X_explain)[0])
        shap_values_explain = explainer(X_explain)
        shap_row = shap_values_explain[0]
    else:
        # Encara no hi ha res a explicar fins que es premi el botó
        X_explain_display = None
        y_explain_pred = None

# -------------------------
# SHOW SELECTED STUDENT + METRIC
# -------------------------
if X_explain_display is not None and y_explain_pred is not None:
    st.write("### Selected student features")
    st.dataframe(X_explain_display)
    st.metric(
        "Predicted dropout risk for this student",
        f"{y_explain_pred * 100:.1f}%",
    )
else:
    st.stop()

# -------------------------
# WATERFALL PLOT
# -------------------------
st.subheader("SHAP Waterfall plot")
st.write(
    """
    The **waterfall plot** shows how each feature contributes to move the prediction
    from the model's **base value** (average prediction) to the final **dropout risk**
    for this student.
    """
)
with st.spinner("Rendering waterfall plot..."):
    plt.figure()
    shap.plots.waterfall(shap_row, show=False)
    fig_w = plt.gcf()
    st.pyplot(fig_w)
    plt.close(fig_w)

# -------------------------
# FORCE PLOT
# -------------------------
st.subheader("SHAP Force plot")
st.write(
    """
    The **force plot** provides a compact view of how features push the prediction
    above (red) or below (blue) the base value for this specific student.
    """
)
with st.spinner("Rendering force plot..."):
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_row.values,
        X_explain.iloc[0, :],
        matplotlib=False,
    )
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(force_html, height=220, scrolling=True)

# -------------------------
# DECISION PLOT
# -------------------------
st.subheader("SHAP Decision plot")
st.write(
    """
    The **decision plot** shows how the model gradually builds the prediction by adding
    feature contributions one by one, from the base value to the final dropout risk
    of this student.
    """
)
with st.spinner("Rendering decision plot..."):
    plt.figure()
    shap.decision_plot(
        base_value=shap_row.base_values,
        shap_values=shap_row.values,
        features=X_explain.iloc[0, :],
        show=False,
    )
    fig_dec = plt.gcf()
    st.pyplot(fig_dec)
    plt.close(fig_dec)
