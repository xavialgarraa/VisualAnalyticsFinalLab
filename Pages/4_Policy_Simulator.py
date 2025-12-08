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
# LOAD MODEL
# -------------------------
try:
    with open("dropout_model.pkl", "rb") as file:
        data = pickle.load(file)
    model_loaded = data["model"]
except FileNotFoundError:
    st.error("Error: 'dropout_model.pkl' not found. Please ensure the trained model file is in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# FEATURES
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
# METADATA
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
# LOAD & ENCODE DATASET
# -------------------------
try:
    df_raw = pd.read_csv("student_dropout.csv")
except FileNotFoundError:
    st.error("Error: 'student_dropout.csv' not found. Please place the dataset file in the directory.")
    st.stop()

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

# -------------------------
# HELPERS
# -------------------------
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


def encode_for_model(answers: dict) -> pd.DataFrame:
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


def encode_and_predict(answers: dict) -> float:
    X_sample = encode_for_model(answers)
    y = model_loaded.predict(X_sample)[0]
    return float(y)

# -------------------------
# BANNED FEATURES
# -------------------------
banned_features = {
    'Gender',
    'Age',
    'Family_Size',
    'Parental_Status',
    'Mother_Education',
    'Father_Education',
    'Mother_Job',
    'Father_Job',
    'Guardian',
    'Family_Support',
    'Family_Relationship',
    'Reason_for_Choosing_School',
}

# -------------------------
# POLICY CATEGORY GROUPS
# -------------------------
policy_categories = {
    "Personal changes": [
        "Study_Time",
        "Number_of_Failures",
        "Number_of_Absences",
        "In_Relationship",
        "Free_Time",
        "Going_Out",
        "Weekend_Alcohol_Consumption",
        "Weekday_Alcohol_Consumption",
        "Health_Status",
        "Wants_Higher_Education",
        "Extra_Curricular_Activities",
    ],
    "Family changes": [
        "Address",
        "Internet_Access",
        "Extra_Paid_Class",
        "Family_Support",
        "Attended_Nursery",
        "Travel_Time",
    ],
    "School changes": [
        "School",
        "School_Support",
    ],
}

# -------------------------
# NUMERIC FEATURES WHERE MAX/MIN IS "GOOD"
# -------------------------
good_high = {
    "Study_Time",
    "Health_Status",
    "Free_Time",
    "Wants_Higher_Education",
}

good_low = {
    "Number_of_Failures",
    "Number_of_Absences",
    "Travel_Time",
    "Weekend_Alcohol_Consumption",
    "Weekday_Alcohol_Consumption",
    "Going_Out",
}

# -------------------------
# SCHOLARSHIP DESCRIPTIONS
# -------------------------
scholarship_details = {
    "School support scholarship": {
        "what": "Financial support to provide structured tutoring or remedial lessons inside the school.",
        "use": "Used to fund extra sessions with teachers or specialised staff so the student receives regular academic support."
    },
    "Digital access scholarship": {
        "what": "A grant to provide a stable internet connection and basic digital equipment at home.",
        "use": "Used to pay internet fees and, if needed, basic devices so the student can access online resources and homework."
    },
    "Private tutoring scholarship": {
        "what": "Coverage of extra paid classes outside regular school hours.",
        "use": "Used to fund private classes or coaching centres focusing on subjects where the student struggles the most."
    },
    "Family education support scholarship": {
        "what": "Programme to support families with guidance, workshops and educational materials.",
        "use": "Used to train and empower parents or guardians so they can better support the student’s learning at home."
    },
}

# -------------------------
# SESSION STATE INITIALISATION
# -------------------------
if "policy_mode" not in st.session_state:
    st.session_state["policy_mode"] = "Use your last predicted student (from Dropout Predictor)"

if "policy_base_student" not in st.session_state:
    st.session_state["policy_base_student"] = None

if "policy_base_risk" not in st.session_state:
    st.session_state["policy_base_risk"] = None

if "policy_source_description" not in st.session_state:
    st.session_state["policy_source_description"] = ""

if "policy_reset_sliders" not in st.session_state:
    st.session_state["policy_reset_sliders"] = False

# -------------------------
# APP TITLE & INTRO
# -------------------------
st.title("🏛️ Policy Impact Simulator")
st.write(
    """
    This page allows you to **simulate concrete policy actions** for a single student and see how
    those actions might change their **predicted dropout risk**.

    - **Step 1:** Select the student profile (from your last prediction, from the dataset, or by entering it manually).  
    - **Step 2:** Explore **Personal**, **Family** and **School** changes and see how each one affects the risk.  
    - **Step 3:** Check which **scholarships** the system would suggest, based on missing support.  
    - **Step 4:** Use the **Recommendations** section to focus on the changes and scholarships that are estimated
      to reduce dropout risk by **more than 10%**.
    """
)
st.divider()

# -------------------------
# 1. STUDENT SELECTION
# -------------------------
st.header("1. Select the student to simulate policies for")

col_m1, col_m2, col_m3 = st.columns(3)

btn_labels = [
    "Use your last predicted student (from Dropout Predictor)",
    "Pick a student from the dataset",
    "Enter a new student manually",
]

with col_m1:
    if st.button("Use last predicted student", use_container_width=True):
        st.session_state["policy_mode"] = btn_labels[0]

with col_m2:
    if st.button("Pick from dataset", use_container_width=True):
        st.session_state["policy_mode"] = btn_labels[1]

with col_m3:
    if st.button("Enter manually", use_container_width=True):
        st.session_state["policy_mode"] = btn_labels[2]

mode = st.session_state["policy_mode"]
st.caption(f"Current selection mode: **{mode}**")

has_last_pred = (
    "explain_X_display" in st.session_state
    and "explain_X_encoded" in st.session_state
    and "explain_y_pred" in st.session_state
    and isinstance(st.session_state["explain_X_encoded"], pd.DataFrame)
    and len(st.session_state["explain_X_encoded"]) > 0
)

new_base_student = None
new_base_risk = None
new_source_desc = ""
updated_base = False

# OPTION 1: LAST PREDICTED
if mode == "Use your last predicted student (from Dropout Predictor)":
    if has_last_pred:
        X_explain_display = st.session_state["explain_X_display"]
        new_base_risk = float(st.session_state["explain_y_pred"])
        new_base_student = X_explain_display.iloc[0].to_dict()
        new_source_desc = (
            "You are simulating policies for the **last student you predicted** "
            "in the Dropout Predictor page."
        )
        updated_base = True
    else:
        st.warning(
            "No previous prediction found from the Dropout Predictor page. "
            "Please predict a student there first, or use one of the other options."
        )

# OPTION 2: DATASET
if mode == "Pick a student from the dataset":
    instance_index = st.slider(
        "Select a student index from the dataset",
        min_value=0,
        max_value=X_test.shape[0] - 1,
        value=0,
        step=1,
        help="Choose which student from the dataset you want to simulate.",
    )
    X_row = X_test.iloc[[instance_index]]

    X_display = X_row.copy()
    for col in cat_mappings:
        try:
            X_display[col] = (
                X_display[col]
                .astype(int)
                .map(
                    lambda idx, cats=cat_mappings[col]: cats[idx]
                    if 0 <= idx < len(cats)
                    else "Unknown"
                )
            )
        except Exception:
            pass

    new_base_risk = float(model_loaded.predict(X_row)[0])
    new_base_student = X_display.iloc[0].to_dict()
    new_source_desc = f"You are simulating policies for **student #{instance_index} from the dataset**."
    updated_base = True

# OPTION 3: MANUAL
if mode == "Enter a new student manually":
    st.info(
        "Fill in the fields below to define a new student profile, then click "
        "**Use this student for policy simulation**."
    )

    with st.form("policy_manual_student_form"):
        # PERSONAL
        st.subheader("👤 Personal information")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p11, col_p22, col_p33, col_p44 = st.columns(4)
        col_p111, col_p222, col_p333, col_p444 = st.columns(4)

        with col_p1:
            gender = input_for_feature("Gender", "pol_man_gender")
            address = input_for_feature("Address", "pol_man_address")

        with col_p2:
            age = input_for_feature("Age", "pol_man_age")

        with col_p11:
            free_time = input_for_feature("Free_Time", "pol_man_free_time")
            in_relationship = input_for_feature("In_Relationship", "pol_man_in_relationship")

        with col_p22:
            going_out = input_for_feature("Going_Out", "pol_man_going_out")

        with col_p111:
            wkday_alc = input_for_feature("Weekday_Alcohol_Consumption", "pol_man_wkday_alc")
            health = input_for_feature("Health_Status", "pol_man_health")

        with col_p222:
            wknd_alc = input_for_feature("Weekend_Alcohol_Consumption", "pol_man_wknd_alc")

        st.markdown("---")

        # FAMILY
        st.subheader("🏠 Family information")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f11, col_f22, col_f33, col_f44 = st.columns(4)

        with col_f1:
            family_size = input_for_feature("Family_Size", "pol_man_family_size")
            parental_status = input_for_feature("Parental_Status", "pol_man_parental_status")

        with col_f2:
            mother_edu = input_for_feature("Mother_Education", "pol_man_mother_edu")
            father_edu = input_for_feature("Father_Education", "pol_man_father_edu")

        with col_f3:
            mother_job = input_for_feature("Mother_Job", "pol_man_mother_job")
            father_job = input_for_feature("Father_Job", "pol_man_father_job")

        with col_f11:
            guardian = input_for_feature("Guardian", "pol_man_guardian")
            family_rel = input_for_feature("Family_Relationship", "pol_man_family_rel")

        st.markdown("---")

        # ACADEMIC
        st.subheader("📚 Academic information")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a11, col_a22, col_a33, col_a44 = st.columns(4)

        with col_a1:
            school = input_for_feature("School", "pol_man_school")
            reason_school = input_for_feature("Reason_for_Choosing_School", "pol_man_reason_school")

        with col_a2:
            travel_time = input_for_feature("Travel_Time", "pol_man_travel_time")
            study_time = input_for_feature("Study_Time", "pol_man_study_time")

        with col_a3:
            num_failures = input_for_feature("Number_of_Failures", "pol_man_num_failures")
            nursery = input_for_feature("Attended_Nursery", "pol_man_nursery")
            higher_ed = input_for_feature("Wants_Higher_Education", "pol_man_higher_ed")

        with col_a11:
            school_support = input_for_feature("School_Support", "pol_man_school_support")
            family_support = input_for_feature("Family_Support", "pol_man_family_support")

        with col_a22:
            extra_paid = input_for_feature("Extra_Paid_Class", "pol_man_extra_paid")
            extra_curr = input_for_feature("Extra_Curricular_Activities", "pol_man_extra_curr")

        with col_a33:
            internet = input_for_feature("Internet_Access", "pol_man_internet")
            absences = input_for_feature("Number_of_Absences", "pol_man_absences")

        submitted = st.form_submit_button("Use this student for policy simulation")

    if submitted:
        manual_inputs = {
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
        new_base_student = manual_inputs
        new_base_risk = encode_and_predict(manual_inputs)
        new_source_desc = "You are simulating policies for a **manually entered student profile**."
        updated_base = True

# UPDATE BASE STUDENT
if updated_base and new_base_student is not None and new_base_risk is not None:
    prev_student = st.session_state["policy_base_student"]
    changed_student = (prev_student != new_base_student)
    st.session_state["policy_base_student"] = new_base_student
    st.session_state["policy_base_risk"] = new_base_risk
    st.session_state["policy_source_description"] = new_source_desc
    if changed_student:
        st.session_state["policy_reset_sliders"] = True

base_student = st.session_state["policy_base_student"]
base_risk = st.session_state["policy_base_risk"]
source_description = st.session_state["policy_source_description"]

if base_student is None or base_risk is None:
    st.info("Please select or define a student profile to start the policy simulation.")
    st.stop()

# Reset sliders when student changes
if st.session_state.get("policy_reset_sliders", False):
    for col in feature_cols:
        key = f"policy_slider_{col}"
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["policy_reset_sliders"] = False

# -------------------------
# SHOW SELECTED STUDENT + BASE RISK
# -------------------------
st.info(source_description)

st.subheader("Selected student profile")
st.dataframe(pd.DataFrame([base_student]))

base_pct = base_risk * 100
st.subheader("Base predicted dropout risk")

if base_risk < 0.3:
    box = st.success
    level = "LOW"
    emoji = "🟢"
elif base_risk < 0.6:
    box = st.warning
    level = "MEDIUM"
    emoji = "🟠"
else:
    box = st.error
    level = "HIGH"
    emoji = "🔴"

box(
    f"""
    Base dropout risk: **{base_pct:.1f}%**  
    Risk level: **{level}** {emoji}
    """
)

st.markdown(
    """
    A higher value means a higher probability that this student will **drop out**.  
    Policy changes below show how modifying a single factor could increase or decrease
    this risk.
    """
)

st.divider()

# -------------------------
# SCHOLARSHIP LOGIC
# -------------------------
def _is_negative(val) -> bool:
    s = str(val).strip().lower()
    return s in {"no", "none", "0", "without", "absent", "nan", "false"}


def compute_scholarships(student: dict, base_risk: float):
    """
    Scholarships are suggested when the student lacks:
    - School support
    - Internet access
    - Extra paid classes
    - Family educational support

    We also estimate the risk reduction if the scholarship is granted.
    """
    scholarships = []

    if "School_Support" in student and _is_negative(student["School_Support"]):
        scholarships.append({
            "Scholarship": "School support scholarship",
            "Feature": "School_Support",
            "Reason": "The student does not receive extra school support."
        })

    if "Internet_Access" in student and _is_negative(student["Internet_Access"]):
        scholarships.append({
            "Scholarship": "Digital access scholarship",
            "Feature": "Internet_Access",
            "Reason": "The student does not have home internet access."
        })

    if "Extra_Paid_Class" in student and _is_negative(student["Extra_Paid_Class"]):
        scholarships.append({
            "Scholarship": "Private tutoring scholarship",
            "Feature": "Extra_Paid_Class",
            "Reason": "The student does not attend extra paid classes."
        })

    if "Family_Support" in student and _is_negative(student["Family_Support"]):
        scholarships.append({
            "Scholarship": "Family education support scholarship",
            "Feature": "Family_Support",
            "Reason": "The family does not provide educational support."
        })

    # Compute impact for each scholarship
    for sch in scholarships:
        feat = sch["Feature"]
        best_new_risk = None
        best_cat = None

        if feat in cat_mappings:
            for cat in cat_mappings[feat]:
                if _is_negative(cat):
                    continue
                ua = student.copy()
                ua[feat] = cat
                r = encode_and_predict(ua)
                if best_new_risk is None or r < best_new_risk:
                    best_new_risk = r
                    best_cat = cat

        if best_new_risk is not None:
            sch["Target_value"] = best_cat
            sch["New_risk"] = best_new_risk
            sch["Risk_change_pct"] = (base_risk - best_new_risk) * 100.0
        else:
            sch["Target_value"] = None
            sch["New_risk"] = None
            sch["Risk_change_pct"] = None

    return scholarships


scholarships = compute_scholarships(base_student, base_risk)

# -------------------------
# 2. POLICY CHANGES
# -------------------------
st.header("2. Policy changes (what-if analysis for each feature)")
st.write(
    """
    Below you can explore policy changes for all **non-banned features**, grouped by who
    would need to act: **Personal changes, Family changes and School changes**.  
    - Each card corresponds to a **policy lever** (support programme, lifestyle change, etc.).  
    - The card shows **what happens if we change this variable** for the selected student.  
    - Moving sliders or testing categories **never modifies the base student profile**; the model
      always uses a **copy** for the predictions.
    """
)


def render_policy_panel(col_name: str, student_base: dict, base_risk: float):
    info = feature_info.get(col_name, {'label': col_name, 'help': ""})
    nice_name = info['label']
    help_text = info['help']
    cur_val = student_base[col_name]

    st.markdown(f"### {nice_name}")
    if help_text:
        st.caption(help_text)

    st.markdown(f"**Student value:** `{cur_val}`")

    # CATEGORICAL
    if col_name in cat_mappings:
        for cat in cat_mappings[col_name]:
            if cat == cur_val:
                continue

            ua = student_base.copy()
            ua[col_name] = cat
            r = encode_and_predict(ua)
            delta = r - base_risk
            risk_pct = r * 100
            delta_pct = delta * 100

            st.markdown(
                f'_If we change "{nice_name}" from **{cur_val}** to **{cat}**, then:_'
            )

            if abs(delta) < 0.001:
                st.metric("New dropout risk", f"{risk_pct:.1f}%")
                st.caption("no meaningful change")
            else:
                st.metric(
                    "New dropout risk",
                    f"{risk_pct:.1f}%",
                    delta=f"{delta_pct:+.1f} %",
                    delta_color="inverse",
                )

        st.markdown("---")
        return

    # NUMERIC
    col_min = float(df_num[col_name].min())
    col_max = float(df_num[col_name].max())
    cur_float = float(cur_val)

    if col_name in good_low:
        initial_default = int(col_min)
    elif col_name in good_high:
        initial_default = int(col_max)
    else:
        initial_default = int(cur_float)

    st.markdown("_Move the slider to simulate a policy change on this variable._")

    new_val = st.slider(
        "New value",
        min_value=int(col_min),
        max_value=int(col_max),
        value=st.session_state.get(f"policy_slider_{col_name}", initial_default),
        step=1,
        key=f"policy_slider_{col_name}",
    )

    st.markdown(
        f'_If we change "{nice_name}" from **{cur_float}** to **{new_val}**, then:_'
    )

    if abs(new_val - cur_float) < 1e-9:
        r = base_risk
        risk_pct = r * 100
        st.metric("New dropout risk", f"{risk_pct:.1f}%")
        st.caption("no change (same value)")
    else:
        ua = student_base.copy()
        ua[col_name] = new_val
        r = encode_and_predict(ua)
        delta = r - base_risk
        risk_pct = r * 100
        delta_pct = delta * 100

        if abs(delta) < 0.001:
            st.metric("New dropout risk", f"{risk_pct:.1f}%")
            st.caption("no meaningful change")
        else:
            st.metric(
                "New dropout risk",
                f"{risk_pct:.1f}%",
                delta=f"{delta_pct:+.1f} %",
                delta_color="inverse",
            )

    st.markdown("---")


for cat_name, cat_feats in policy_categories.items():
    if cat_name == "Personal changes":
        st.markdown("## 🧍 Personal changes")
    elif cat_name == "Family changes":
        st.markdown("## 🏠 Family changes")
    elif cat_name == "School changes":
        st.markdown("## 🏫 School changes")
    else:
        st.subheader(cat_name)

    cols = st.columns(4)
    visible_feats = [f for f in cat_feats if f not in banned_features]
    for idx, feat in enumerate(visible_feats):
        with cols[idx % 4]:
            render_policy_panel(feat, base_student, base_risk)

# ---------- DIVIDER BEFORE SCHOLARSHIPS ----------
st.divider()

# -------------------------
# 3. Scholarships
# -------------------------
st.header("3. Scholarships")
if not scholarships:
    st.info(
        "No specific scholarship recommendation is triggered for this profile based on support variables. "
        "The student already has school support, internet, extra paid classes and family educational support."
    )
else:
    st.write(
        """
        Based on the current profile, the following **scholarships** are recommended because the student
        is missing key forms of support (`school support`, `internet`, `extra paid classes` or `family educational support`).  
        Each card below explains **what the scholarship is**, **how it would be used**, and a qualitative
        indication of its **potential impact** on dropout risk.
        """
    )
    n = len(scholarships)
    n_cols = 2 if n > 1 else 1
    cols = st.columns(n_cols)

    for i, sch in enumerate(scholarships):
        col = cols[i % n_cols]
        with col:
            name = sch["Scholarship"]
            desc = scholarship_details.get(name, {})
            what = desc.get("what", "Scholarship programme aimed at reducing structural barriers.")
            use = desc.get("use", "Used to support the student with targeted resources and services.")

            st.markdown(f"### 🎓 {name}")
            st.markdown(f"**Why recommended:** {sch['Reason']}")
            st.markdown(f"**What it is:** {what}")
            st.markdown(f"**How it is used:** {use}")

            rc = sch.get("Risk_change_pct")

            if rc is not None and rc > 0:
                st.markdown(
                    f"**Estimated impact on dropout risk:** this scholarship could reduce the risk "
                    f"by approximately **{rc:.1f}%** for this student."
                )
            elif rc is not None and rc <= 0:
                st.markdown(
                    f"**Estimated impact on dropout risk:** no clear reduction is expected based on the model "
                    f"for this student."
                )
            else:
                st.markdown(
                    "**Estimated impact on dropout risk:** not enough data to estimate the effect."
                )

# -------------------------
# 4. Recommendations (changes & scholarships with impact > 10%)
# -------------------------
st.divider()
st.header("4. Recommendations")
st.write(
    """
    This section highlights **policy changes and scholarships** that are estimated to reduce
    dropout risk by **more than 10%** (absolute reduction) for this student.
    """
)

recommendation_cards = []

# --- Scholarships with strong impact (risk reduction > 10%) ---
for sch in scholarships:
    rc = sch.get("Risk_change_pct")
    if rc is not None and rc > 10.0:
        recommendation_cards.append(
            {
                "type": "scholarship",
                "obj": sch,
            }
        )

# --- Policy changes: simulate all concrete changes and keep those with risk reduction > 10% ---
summary_rows = []

for col_name in feature_cols:
    if col_name in banned_features:
        continue

    feat_label = feature_info.get(col_name, {}).get("label", col_name)
    cur_val = base_student[col_name]
    cur_risk = base_risk

    # CATEGORICAL FEATURES
    if col_name in cat_mappings:
        for cat in cat_mappings[col_name]:
            if cat == cur_val:
                continue

            ua = base_student.copy()
            ua[col_name] = cat
            new_risk = encode_and_predict(ua)
            change_pct = (cur_risk - new_risk) * 100.0  # positive = risk reduction

            if change_pct > 10.0:
                summary_rows.append(
                    {
                        "Feature": feat_label,
                        "Change": f'{feat_label}: {cur_val} → {cat}',
                        "Risk change (%)": change_pct,
                        "New_risk": new_risk,
                    }
                )

    # NUMERIC FEATURES
    else:
        col_min = float(df_num[col_name].min())
        col_max = float(df_num[col_name].max())
        cur_float = float(cur_val)

        targets = []

        if col_name in good_low:
            targets.append(("min (good)", col_min))

        if col_name in good_high:
            targets.append(("max (good)", col_max))

        if not targets:
            continue

        for tag, target in targets:
            if abs(target - cur_float) < 1e-9:
                continue

            ua = base_student.copy()
            ua[col_name] = target
            new_risk = encode_and_predict(ua)
            change_pct = (cur_risk - new_risk) * 100.0

            if change_pct > 10.0:
                summary_rows.append(
                    {
                        "Feature": feat_label,
                        "Change": f'{feat_label}: {cur_float:.1f} → {target:.1f} ({tag})',
                        "Risk change (%)": change_pct,
                        "New_risk": new_risk,
                    }
                )

# Add policy changes to recommendation cards
for row in summary_rows:
    recommendation_cards.append(
        {
            "type": "change",
            "obj": row,
        }
    )

if not recommendation_cards:
    st.info(
        "There are no scholarships or policy changes with an estimated dropout risk reduction above 10% for this student."
    )
else:
    st.write(
        """
        These **scholarships and policy changes** are estimated to have a **strong impact**
        (more than **10%** risk reduction).  
        Each card shows the recommended action and the new predicted dropout risk.
        """
    )
    n = len(recommendation_cards)
    n_cols = 2 if n > 1 else 1
    cols = st.columns(n_cols)

    for i, rec in enumerate(recommendation_cards):
        col = cols[i % n_cols]
        with col:
            if rec["type"] == "scholarship":
                sch = rec["obj"]
                name = sch["Scholarship"]
                desc = scholarship_details.get(name, {})
                what = desc.get("what", "Scholarship programme aimed at reducing structural barriers.")
                use = desc.get("use", "Used to support the student with targeted resources and services.")
                new_risk = sch.get("New_risk", base_risk)
                base_pct = base_risk * 100.0
                new_pct = new_risk * 100.0
                delta_pct = new_pct - base_pct

                st.markdown(f"### 🎯 Scholarship recommendation: {name}")
                st.markdown(f"**Why recommended:** {sch['Reason']}")
                st.markdown(f"**What it is:** {what}")
                st.markdown(f"**How it is used:** {use}")
                st.metric(
                    "Estimated dropout risk after scholarship",
                    f"{new_pct:.1f}%",
                    delta=f"{delta_pct:+.1f} %",
                    delta_color="inverse",
                )

            else:
                row = rec["obj"]
                change_desc = row["Change"]
                new_risk = row["New_risk"]
                base_pct = base_risk * 100.0
                new_pct = new_risk * 100.0
                delta_pct = new_pct - base_pct

                st.markdown("### 🔧 Policy change recommendation")
                st.markdown(f"**Change:** {change_desc}")
                st.markdown(
                    "This represents a concrete modification in the student's context or behaviour "
                    "that the school, the family or the student could implement."
                )
                st.metric(
                    "Estimated dropout risk after this change",
                    f"{new_pct:.1f}%",
                    delta=f"{delta_pct:+.1f} %",
                    delta_color="inverse",
                )
