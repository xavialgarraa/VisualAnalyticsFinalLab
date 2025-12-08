import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pandas.api.types import is_numeric_dtype

st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
)

with open('dropout_model.pkl', 'rb') as file:
    data = pickle.load(file)

model_loaded = data["model"]

feature_cols = [
    'School',
    'Gender',
    'Age',
    'Address',
    'Family_Size',
    'Parental_Status',
    'Mother_Education',
    'Father_Education',
    'Mother_Job',
    'Father_Job',
    'Reason_for_Choosing_School',
    'Guardian',
    'Travel_Time',
    'Study_Time',
    'Number_of_Failures',
    'School_Support',
    'Family_Support',
    'Extra_Paid_Class',
    'Extra_Curricular_Activities',
    'Attended_Nursery',
    'Wants_Higher_Education',
    'Internet_Access',
    'In_Relationship',
    'Family_Relationship',
    'Free_Time',
    'Going_Out',
    'Weekend_Alcohol_Consumption',
    'Weekday_Alcohol_Consumption',
    'Health_Status',
    'Number_of_Absences'
]

# METADADES DE FEATURES (labels + help)
feature_info = {
    'School': {'label': "School", 'help': None},
    'Gender': {'label': "Gender", 'help': "M for Male and F for Female."},
    'Age': {'label': "Age", 'help': "Age of the student."},
    'Address': {'label': "Address", 'help': "U for urban and R for rural."},
    'Family_Size': {'label': "Family size", 'help': "GT3 for >3 and LE3 for ≤3."},
    'Parental_Status': {'label': "Parental status", 'help': "A for together and T for apart."},
    'Mother_Education': {'label': "Mother education level", 'help': None},
    'Father_Education': {'label': "Father education level", 'help': None},
    'Mother_Job': {'label': "Mother job", 'help': None},
    'Father_Job': {'label': "Father job", 'help': None},
    'Reason_for_Choosing_School': {'label': "Reason for choosing school", 'help': None},
    'Guardian': {'label': "Guardian", 'help': None},
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

def input_for_feature(col_name: str, key: str):
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

def encode_and_predict(answers: dict) -> float:
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
    return float(model_loaded.predict(X_sample)[0])

button_labels = {
    'School': "Change the student's school",
    'Address': "Change the student's address",
    'Travel_Time': "Change the travel time to school",
    'Study_Time': "Change weekly study time",
    'Number_of_Failures': "Change number of past failures",
    'School_Support': "Change school support",
    'Extra_Paid_Class': "Change extra paid classes",
    'Extra_Curricular_Activities': "Change extracurricular activities",
    'Attended_Nursery': "Change nursery attendance",
    'Wants_Higher_Education': "Change higher education intention",
    'Internet_Access': "Change internet access",
    'In_Relationship': "Change relationship status",
    'Free_Time': "Change free time",
    'Going_Out': "Change going out frequency",
    'Weekend_Alcohol_Consumption': "Change weekend alcohol consumption",
    'Weekday_Alcohol_Consumption': "Change weekday alcohol consumption",
    'Health_Status': "Change health status",
    'Number_of_Absences': "Change number of absences"
}

st.title("Student Dropout Risk Predictor 🎓")
st.write("""Welcome to the Student Dropout Risk section. Enter the following Personal, 
            Familiar and Academic information about the student to estimate his/her dropout risk.""")
st.write("""After predicting the dropout risk, you will see a set of **What-If buttons**.  
Each button lets you test the effect of changing one specific feature of the student profile.  
- Clicking a button opens a small panel right below the button.
- For **categorical features**, you will see alternative options and how each option affects the predicted dropout risk.
- For **numeric features**, you will get a slider where you can explore different values and immediately view how the prediction changes.
- The effect is shown using **“New dropout risk”**, along with the difference from the original prediction:
  - Green = risk decreases  
  - Red = risk increases  
  - If the change is extremely small (less than 0.001), it is marked as "no change".   

This allows you to explore how each individual factor influences dropout risk, without altering the original student profile.
""")

# PERSONAL
st.header("👤 Personal information")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p11, col_p22, col_p33, col_p44 = st.columns(4)
col_p111, col_p222, col_p333, col_p444 = st.columns(4)

with col_p1:
    gender = input_for_feature("Gender", "pp_gender")
    address = input_for_feature("Address", "pp_address")

with col_p2:
    age = input_for_feature("Age", "pp_age")

with col_p11:
    free_time = input_for_feature("Free_Time", "pp_free_time")
    in_relationship = input_for_feature("In_Relationship", "pp_in_relationship")

with col_p22:
    going_out = input_for_feature("Going_Out", "pp_going_out")

with col_p111:
    wkday_alc = input_for_feature("Weekday_Alcohol_Consumption", "pp_wkday_alc")
    health = input_for_feature("Health_Status", "pp_health")

with col_p222:
    wknd_alc = input_for_feature("Weekend_Alcohol_Consumption", "pp_wknd_alc")

st.markdown("---")

# FAMILY
st.header("🏠 Family information")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
col_f11, col_f22, col_f33, col_f44 = st.columns(4)

with col_f1:
    family_size = input_for_feature("Family_Size", "pp_family_size")
    parental_status = input_for_feature("Parental_Status", "pp_parental_status")
    guardian = input_for_feature("Guardian", "pp_guardian")

with col_f11:
    mother_job = input_for_feature("Mother_Job", "pp_mother_job")
    father_job = input_for_feature("Father_Job", "pp_father_job")
    family_support = input_for_feature("Family_Support", "pp_family_support")
    
with col_f22:
    mother_edu = input_for_feature("Mother_Education", "pp_mother_edu")
    father_edu = input_for_feature("Father_Education", "pp_father_edu")
    fam_rel = input_for_feature("Family_Relationship", "pp_fam_rel")

st.markdown("---")

# ACADEMIC
st.header("🏫 Academic information")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    school = input_for_feature("School", "pp_school")
    school_support = input_for_feature("School_Support", "pp_school_support")
    travel_time = input_for_feature("Travel_Time", "pp_travel_time")
    absences = input_for_feature("Number_of_Absences", "pp_absences")
    internet = input_for_feature("Internet_Access", "pp_internet")
    extra_curr = input_for_feature("Extra_Curricular_Activities", "pp_extra_curr")
    
with col_s2:
    reason_school = input_for_feature("Reason_for_Choosing_School", "pp_reason_school")
    extra_paid = input_for_feature("Extra_Paid_Class", "pp_extra_paid")
    study_time = input_for_feature("Study_Time", "pp_study_time")
    num_failures = input_for_feature("Number_of_Failures", "pp_num_failures")
    nursery = input_for_feature("Attended_Nursery", "pp_nursery")
    higher_ed = input_for_feature("Wants_Higher_Education", "pp_higher_ed")
    
st.divider()

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
    'Family_Relationship': fam_rel,
    'Free_Time': free_time,
    'Going_Out': going_out,
    'Weekend_Alcohol_Consumption': wknd_alc,
    'Weekday_Alcohol_Consumption': wkday_alc,
    'Health_Status': health,
    'Number_of_Absences': absences
}

last_inputs = st.session_state.get("last_inputs")
if last_inputs is not None and last_inputs != current_inputs:
    for key in [
        "base_risk",
        "base_answers",
        "whatif_feature",
        "explain_X_display",
        "explain_X_encoded",
        "explain_y_pred",
    ]:
        st.session_state.pop(key, None)

# BOTÓ DE PREDICCIÓ
if st.button("Predict Dropout Risk"):
    try:
        user_answers = current_inputs.copy()

        y_pred = encode_and_predict(user_answers)

        st.session_state["base_answers"] = user_answers
        st.session_state["base_risk"] = y_pred
        st.session_state["whatif_feature"] = None
        st.session_state["last_inputs"] = current_inputs

        sample_list_base = []
        for col in feature_cols:
            v = user_answers[col]
            if col in cat_mappings:
                cats = cat_mappings[col]
                c = cats.index(v)
                sample_list_base.append(float(c))
            else:
                sample_list_base.append(float(v))
        st.session_state["explain_X_display"] = pd.DataFrame([user_answers])
        st.session_state["explain_X_encoded"] = pd.DataFrame([sample_list_base], columns=feature_cols)
        st.session_state["explain_y_pred"] = float(y_pred)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# RESULTATS + WHAT-IF 
if "base_risk" in st.session_state and "base_answers" in st.session_state:
    y_pred = st.session_state["base_risk"]
    user_answers = st.session_state["base_answers"]

    base_pct = y_pred * 100

    st.subheader("Predicted Dropout Risk")
    if y_pred < 0.3:
        box = st.success
        level = "LOW"
        emoji = "🟢"
    elif y_pred < 0.6:
        box = st.warning
        level = "MEDIUM"
        emoji = "🟠"
    else:
        box = st.error
        level = "HIGH"
        emoji = "🔴"

    box(f"""
        Student dropout risk: **{base_pct:.1f}%**  
        Risk level: **{level}** {emoji}
        """)

    st.markdown("### 🤔 What-if we...")

    banned_features = {
        'Gender', 'Age',
        'Family_Size', 'Parental_Status', 'Mother_Education',
        'Father_Education', 'Mother_Job', 'Father_Job',
        'Guardian', 'Family_Support', 'Family_Relationship',
        'Reason_for_Choosing_School'
    }

    col_b1, col_b2 = st.columns(2)
    button_cols = [col_b1, col_b2]

    containers = {}
    clicked_feature = None

    for idx, col in enumerate(feature_cols):
        if col in banned_features:
            continue

        btn_label = button_labels.get(col, f"Change {col}")
        parent_col = button_cols[idx % 2]

        with parent_col:
            cont = st.container()
            containers[col] = cont
            with cont:
                if st.button(btn_label, key=f"btn_{col}"):
                    clicked_feature = col

    if clicked_feature is not None:
        st.session_state["whatif_feature"] = clicked_feature

    active_feature = st.session_state.get("whatif_feature")

    if active_feature is not None and active_feature in containers:
        col = active_feature
        cur_val = user_answers[col]
        nice_name = feature_info.get(col, {}).get("label", col)

        with containers[col]:
            # CATEGÒRIQUES
            if col in cat_mappings:
                for cat in cat_mappings[col]:
                    if cat == cur_val:
                        continue

                    ua = user_answers.copy()
                    ua[col] = cat
                    r = encode_and_predict(ua)
                    delta = r - y_pred

                    risk_pct = r * 100
                    delta_pct = delta * 100

                    if col == "School":
                        desc = f"Changing the student from {cur_val} School to {cat} School"
                    else:
                        desc = f"Changing {nice_name} from {cur_val} to {cat}"

                    st.write(desc)

                    if abs(delta) < 0.001:
                        st.metric("New dropout risk", f"{risk_pct:.1f}%")
                        st.caption("no change")
                    else:
                        st.metric(
                            "New dropout risk",
                            f"{risk_pct:.1f}%",
                            delta=f"{delta_pct:+.1f} %",
                            delta_color="inverse"
                        )

            # NUMÈRIQUES
            else:
                cur_float = float(cur_val)
                col_min = float(df_num[col].min())
                col_max = float(df_num[col].max())

                new_val = st.slider(
                    f"New value for {nice_name}",
                    min_value=int(col_min),
                    max_value=int(col_max),
                    value=st.session_state.get(f"whatif_slider_{col}", int(cur_float)),
                    step=1,
                    key=f"whatif_slider_{col}"
                )

                desc = (
                    f"Changing {nice_name} from {cur_float:.1f} "
                    f"to {new_val:.1f}"
                )
                st.write(desc)

                metric_col, _ = st.columns([1, 2])

                with metric_col:
                    if abs(new_val - cur_float) < 1e-9:
                        r = y_pred
                        risk_pct = r * 100
                        st.metric("New dropout risk", f"{risk_pct:.1f}%")
                        st.caption("no change")
                    else:
                        ua = user_answers.copy()
                        ua[col] = new_val
                        r = encode_and_predict(ua)
                        delta = r - y_pred

                        risk_pct = r * 100
                        delta_pct = delta * 100

                        if abs(delta) < 0.001:
                            st.metric("New dropout risk", f"{risk_pct:.1f}%")
                            st.caption("no change")
                        else:
                            st.metric(
                                "New dropout risk",
                                f"{risk_pct:.1f}%",
                                delta=f"{delta_pct:+.1f} %",
                                delta_color="inverse"
                            )
    st.markdown(
        """
        <div style="
            background-color: #e7f0ff;
            padding: 18px;
            border-radius: 8px;
            border-left: 5px solid #5b9bff;
            font-size: 16px;
            ">
            <strong>Want to explore deeper?</strong><br>
            You can use the <em>Changes Impact Simulator</em> to analyse how 
            different policies, interventions, or lifestyle adjustments could influence 
            this student's dropout risk.  
            It provides a more advanced scenario analysis with grouped changes, 
            scholarships, and recommendations based on predicted impact.
            <br><br>
            👉 <strong>Go to the “Changes Impact Simulator” page to investigate more possibilities.</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("Set the student profile and click **Predict Dropout Risk** to see what-if analysis.")
