import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load and preprocess data
# -------------------------
df = pd.read_csv("student_dropout.csv")

# Drop grade columns as requested
df = df.drop(columns=["Grade_1", "Grade_2", "Final_Grade"])

# Drop missing values (same behaviour as original)
df = df.dropna()

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

st.set_page_config(
    page_title="Student Dropout – Data Explorer",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Dropout Data Explorer 🔎📊")
st.write(
    "Welcome to the student dropout data exploration section. This page helps you understand how "
    "different student characteristics relate to **dropout (Dropped_Out)**."
)
st.divider()

st.subheader("Filter the dataset")
st.write(
    """In the sidebar you can filter the dataset as you wish to explore the specific students you want.
    All the values and plots will update dynamically based on your selected filters. 
    The default state includes all the data."""
)

st.sidebar.header("Filters")

# -------------------------
# Helpers
# -------------------------
def radio_three(label, options, key, help_text=""):
    options = sorted(options)
    selected = st.sidebar.radio(label, ["Both"] + options, key=key, help=help_text)
    if selected == "Both":
        return options
    return [selected]

def radio_yes_no_both(label, column_name, key_prefix, help_text=""):
    values = sorted(df[column_name].unique())
    choice = st.sidebar.radio(
        label,
        ["Both"] + list(values),
        index=0,
        key=f"{key_prefix}_radio",
        help=help_text
    )
    if choice == "Both":
        return values
    else:
        return [choice]

# -------------------------
# Sidebar filters (radiobuttons)
# -------------------------

# Categorical – radio with Both
selected_schools = radio_three(
    "School",
    df["School"].unique(),
    "school_filter",
    help_text="Name of the school attended."
)
selected_genders = radio_three(
    "Gender",
    df["Gender"].unique(),
    "gender_filter",
    help_text="M for Male and F for Female)."
)
selected_addresses = radio_three(
    "Address (U/R)",
    df["Address"].unique(),
    "address_filter",
    help_text="U for urban and R for rural."
)
selected_family_sizes = radio_three(
    "Family size",
    df["Family_Size"].unique(),
    "family_size_filter",
    help_text="GT3 for >3 members and LE3 for ≤3 members."
)
selected_parental_status = radio_three(
    "Parental status",
    df["Parental_Status"].unique(),
    "parental_status_filter",
    help_text="A for together and T for apart."
)

# Yes/No filters (existing)
selected_school_support = radio_yes_no_both(
    "School support",
    "School_Support",
    "school_support",
    help_text="Whether the student receives extra school educational support."
)
selected_family_support = radio_yes_no_both(
    "Family support",
    "Family_Support",
    "family_support",
    help_text="Family provided educational support."
)
selected_internet_access = radio_yes_no_both(
    "Internet access",
    "Internet_Access",
    "internet_access",
    help_text="Availability of internet at home."
)
selected_higher_edu = radio_yes_no_both(
    "Wants higher education",
    "Wants_Higher_Education",
    "higher_edu",
    help_text="Desire to pursue higher education."
)
selected_relationship = radio_yes_no_both(
    "In relationship",
    "In_Relationship",
    "relationship",
    help_text="Involved in a romantic relationship."
)

# Optional filter on target (Dropped_Out)
selected_dropout_status = radio_yes_no_both(
    "Dropout status (Dropped_Out)",
    "Dropped_Out",
    "dropout_status",
    help_text="Indicator of whether the student has dropped out."
)

# Extra paid, extra activities, nursery as radio (just after previous radios)
selected_extra_paid = radio_yes_no_both(
    "Extra paid class",
    "Extra_Paid_Class",
    "extra_paid",
    help_text="Participation in extra paid classes."
)
selected_extra_act = radio_yes_no_both(
    "Extra curricular activities",
    "Extra_Curricular_Activities",
    "extra_act",
    help_text="Involvement in extracurricular activities."
)
selected_nursery = radio_yes_no_both(
    "Attended nursery",
    "Attended_Nursery",
    "nursery",
    help_text="Attendance in nursery school."
)

# -------------------------
# Numeric sliders
# -------------------------
age_min, age_max = st.sidebar.slider(
    "Age range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max())),
    help="Age of the student."
)

abs_min, abs_max = st.sidebar.slider(
    "Number of absences range",
    int(df["Number_of_Absences"].min()),
    int(df["Number_of_Absences"].max()),
    (int(df["Number_of_Absences"].min()), int(df["Number_of_Absences"].max())),
    help="Total number of absences from school."
)

fail_min, fail_max = st.sidebar.slider(
    "Number of failures range",
    int(df["Number_of_Failures"].min()),
    int(df["Number_of_Failures"].max()),
    (int(df["Number_of_Failures"].min()), int(df["Number_of_Failures"].max())),
    help="Number of past class failures."
)

study_min, study_max = st.sidebar.slider(
    "Weekly study time (1–4)",
    int(df["Study_Time"].min()),
    int(df["Study_Time"].max()),
    (int(df["Study_Time"].min()), int(df["Study_Time"].max())),
    help="Weekly study hours (1 to 4)."
)

# NEW NUMERIC FILTERS (remaining numeric features)
mother_edu_min, mother_edu_max = st.sidebar.slider(
    "Mother's education (0–4)",
    int(df["Mother_Education"].min()),
    int(df["Mother_Education"].max()),
    (int(df["Mother_Education"].min()), int(df["Mother_Education"].max())),
    help="Education level of the mother (0 to 4)."
)

father_edu_min, father_edu_max = st.sidebar.slider(
    "Father's education (0–4)",
    int(df["Father_Education"].min()),
    int(df["Father_Education"].max()),
    (int(df["Father_Education"].min()), int(df["Father_Education"].max())),
    help="Education level of the father (0 to 4)."
)

travel_min, travel_max = st.sidebar.slider(
    "Travel time to school (minutes)",
    int(df["Travel_Time"].min()),
    int(df["Travel_Time"].max()),
    (int(df["Travel_Time"].min()), int(df["Travel_Time"].max())),
    help="Time taken to travel to school (in minutes)."
)

fam_rel_min, fam_rel_max = st.sidebar.slider(
    "Family relationship quality (1–5)",
    int(df["Family_Relationship"].min()),
    int(df["Family_Relationship"].max()),
    (int(df["Family_Relationship"].min()), int(df["Family_Relationship"].max())),
    help="Quality of family relationships (scale 1 to 5)."
)

free_time_min, free_time_max = st.sidebar.slider(
    "Free time after school (1–5)",
    int(df["Free_Time"].min()),
    int(df["Free_Time"].max()),
    (int(df["Free_Time"].min()), int(df["Free_Time"].max())),
    help="Amount of free time after school (scale 1 to 5)."
)

going_out_min, going_out_max = st.sidebar.slider(
    "Going out with friends (1–5)",
    int(df["Going_Out"].min()),
    int(df["Going_Out"].max()),
    (int(df["Going_Out"].min()), int(df["Going_Out"].max())),
    help="Frequency of going out with friends (scale 1 to 5)."
)

walc_min, walc_max = st.sidebar.slider(
    "Weekend alcohol consumption (1–5)",
    int(df["Weekend_Alcohol_Consumption"].min()),
    int(df["Weekend_Alcohol_Consumption"].max()),
    (int(df["Weekend_Alcohol_Consumption"].min()), int(df["Weekend_Alcohol_Consumption"].max())),
    help="Alcohol consumption on weekends (scale 1 to 5)."
)

dalc_min, dalc_max = st.sidebar.slider(
    "Weekday alcohol consumption (1–5)",
    int(df["Weekday_Alcohol_Consumption"].min()),
    int(df["Weekday_Alcohol_Consumption"].max()),
    (int(df["Weekday_Alcohol_Consumption"].min()), int(df["Weekday_Alcohol_Consumption"].max())),
    help="Alcohol consumption on weekdays (scale 1 to 5)."
)

health_min, health_max = st.sidebar.slider(
    "Health status (1–5)",
    int(df["Health_Status"].min()),
    int(df["Health_Status"].max()),
    (int(df["Health_Status"].min()), int(df["Health_Status"].max())),
    help="Health rating of the student (scale 1 to 5)."
)

# -------------------------
# CATEGORICAL FILTERS (remaining – multiselect)
# -------------------------
mother_job_options = sorted(df["Mother_Job"].unique())
selected_mother_jobs = st.sidebar.multiselect(
    "Mother's job",
    mother_job_options,
    default=mother_job_options,
    help="Type of job held by the mother."
)

father_job_options = sorted(df["Father_Job"].unique())
selected_father_jobs = st.sidebar.multiselect(
    "Father's job",
    father_job_options,
    default=father_job_options,
    help="Type of job held by the father."
)

reason_options = sorted(df["Reason_for_Choosing_School"].unique())
selected_reasons = st.sidebar.multiselect(
    "Reason for choosing school",
    reason_options,
    default=reason_options,
    help="Reason for selecting the school."
)

guardian_options = sorted(df["Guardian"].unique())
selected_guardians = st.sidebar.multiselect(
    "Guardian",
    guardian_options,
    default=guardian_options,
    help="Guardian of the student."
)

# -------------------------
# Outlier option
# -------------------------
outlier_option = st.sidebar.radio(
    "Outliers",
    ["Included", "Excluded"],
    index=0,
    help="If excluded, extreme values of Age and Number_of_Absences will be removed."
)

# -------------------------
# Apply filters
# -------------------------
f = df[
    (df["School"].isin(selected_schools)) &
    (df["Gender"].isin(selected_genders)) &
    (df["Address"].isin(selected_addresses)) &
    (df["Family_Size"].isin(selected_family_sizes)) &
    (df["Parental_Status"].isin(selected_parental_status)) &
    (df["School_Support"].isin(selected_school_support)) &
    (df["Family_Support"].isin(selected_family_support)) &
    (df["Internet_Access"].isin(selected_internet_access)) &
    (df["Wants_Higher_Education"].isin(selected_higher_edu)) &
    (df["In_Relationship"].isin(selected_relationship)) &
    (df["Dropped_Out"].isin(selected_dropout_status)) &
    (df["Extra_Paid_Class"].isin(selected_extra_paid)) &
    (df["Extra_Curricular_Activities"].isin(selected_extra_act)) &
    (df["Attended_Nursery"].isin(selected_nursery)) &
    (df["Age"] >= age_min) & (df["Age"] <= age_max) &
    (df["Number_of_Absences"] >= abs_min) & (df["Number_of_Absences"] <= abs_max) &
    (df["Number_of_Failures"] >= fail_min) & (df["Number_of_Failures"] <= fail_max) &
    (df["Study_Time"] >= study_min) & (df["Study_Time"] <= study_max) &
    (df["Mother_Education"] >= mother_edu_min) & (df["Mother_Education"] <= mother_edu_max) &
    (df["Father_Education"] >= father_edu_min) & (df["Father_Education"] <= father_edu_max) &
    (df["Travel_Time"] >= travel_min) & (df["Travel_Time"] <= travel_max) &
    (df["Family_Relationship"] >= fam_rel_min) & (df["Family_Relationship"] <= fam_rel_max) &
    (df["Free_Time"] >= free_time_min) & (df["Free_Time"] <= free_time_max) &
    (df["Going_Out"] >= going_out_min) & (df["Going_Out"] <= going_out_max) &
    (df["Weekend_Alcohol_Consumption"] >= walc_min) & (df["Weekend_Alcohol_Consumption"] <= walc_max) &
    (df["Weekday_Alcohol_Consumption"] >= dalc_min) & (df["Weekday_Alcohol_Consumption"] <= dalc_max) &
    (df["Health_Status"] >= health_min) & (df["Health_Status"] <= health_max) &
    (df["Mother_Job"].isin(selected_mother_jobs)) &
    (df["Father_Job"].isin(selected_father_jobs)) &
    (df["Reason_for_Choosing_School"].isin(selected_reasons)) &
    (df["Guardian"].isin(selected_guardians))
]

if outlier_option == "Excluded" and len(f) > 0:
    for col in ["Age", "Number_of_Absences"]:
        q_low = f[col].quantile(0.05)
        q_high = f[col].quantile(0.95)
        f = f[(f[col] >= q_low) & (f[col] <= q_high)]

if len(f) == 0:
    st.warning(
        "No data available for the selected filters.\n\n"
        "Try relaxing some filters (e.g., School, Gender, Age or absences range)."
    )
    st.stop()

# -------------------------
# Overview metrics
# -------------------------
st.subheader("Overview of the filtered dataset")

# Helper for percentages
def pct(series, value):
    return series.value_counts(normalize=True).get(value, 0.0) * 100

n_students = len(f)
dropout_rate = float(f["Dropped_Out"].mean() * 100)

# Pre-compute categorical majority values + percentages
top_school = f["School"].value_counts().idxmax()
top_school_pct = pct(f["School"], top_school)

top_gender = f["Gender"].value_counts().idxmax()
top_gender_pct = pct(f["Gender"], top_gender)

top_address = f["Address"].value_counts().idxmax()
top_address_pct = pct(f["Address"], top_address)

top_family_size = f["Family_Size"].value_counts().idxmax()
top_family_size_pct = pct(f["Family_Size"], top_family_size)

top_parental_status = f["Parental_Status"].value_counts().idxmax()
top_parental_status_pct = pct(f["Parental_Status"], top_parental_status)

top_mother_job = f["Mother_Job"].value_counts().idxmax()
top_mother_job_pct = pct(f["Mother_Job"], top_mother_job)

top_father_job = f["Father_Job"].value_counts().idxmax()
top_father_job_pct = pct(f["Father_Job"], top_father_job)

top_reason = f["Reason_for_Choosing_School"].value_counts().idxmax()
top_reason_pct = pct(f["Reason_for_Choosing_School"], top_reason)

top_guardian = f["Guardian"].value_counts().idxmax()
top_guardian_pct = pct(f["Guardian"], top_guardian)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Students (rows)", f"{n_students:,}")
    st.metric("Dropout rate", f"{dropout_rate:.2f}%")
    st.metric("Most common gender", f"{top_gender} ({top_gender_pct:.1f}%)")
    st.metric("Average age", f"{float(f['Age'].mean()):.2f}")
    st.metric("Average absences", f"{float(f['Number_of_Absences'].mean()):.2f}")
    st.metric("Avg travel time (min)", f"{float(f['Travel_Time'].mean()):.2f}")
    st.metric("Avg failures", f"{float(f['Number_of_Failures'].mean()):.2f}")
    st.metric("Avg study time (1–4)", f"{float(f['Study_Time'].mean()):.2f}")

with col2:
    st.metric("Most common school", f"{top_school} ({top_school_pct:.1f}%)")
    st.metric("Most common address", f"{top_address} ({top_address_pct:.1f}%)")
    st.metric("Most common family size", f"{top_family_size} ({top_family_size_pct:.1f}%)")
    st.metric("Most common parental status", f"{top_parental_status} ({top_parental_status_pct:.1f}%)")
    st.metric("Most common mother job", f"{top_mother_job} ({top_mother_job_pct:.1f}%)")
    st.metric("Most common father job", f"{top_father_job} ({top_father_job_pct:.1f}%)")
    st.metric("Most common reason", f"{top_reason} ({top_reason_pct:.1f}%)")
    st.metric("Most common guardian", f"{top_guardian} ({top_guardian_pct:.1f}%)")

with col3:
    st.metric("School support = yes", f"{pct(f['School_Support'], 'yes'):.2f}%")
    st.metric("Family support = yes", f"{pct(f['Family_Support'], 'yes'):.2f}%")
    st.metric("Extra paid class = yes", f"{pct(f['Extra_Paid_Class'], 'yes'):.2f}%")
    st.metric("Extra activities = yes", f"{pct(f['Extra_Curricular_Activities'], 'yes'):.2f}%")
    st.metric("Attended nursery = yes", f"{pct(f['Attended_Nursery'], 'yes'):.2f}%")
    st.metric("Wants higher edu = yes", f"{pct(f['Wants_Higher_Education'], 'yes'):.2f}%")
    st.metric("Internet access = yes", f"{pct(f['Internet_Access'], 'yes'):.2f}%")
    st.metric("In relationship = yes", f"{pct(f['In_Relationship'], 'yes'):.2f}%")

with col4:
    st.metric("Avg mother education (0–4)", f"{float(f['Mother_Education'].mean()):.2f}")
    st.metric("Avg father education (0–4)", f"{float(f['Father_Education'].mean()):.2f}")
    st.metric("Avg family relationship (1–5)", f"{float(f['Family_Relationship'].mean()):.2f}")
    st.metric("Avg free time (1–5)", f"{float(f['Free_Time'].mean()):.2f}")
    st.metric("Avg going out (1–5)", f"{float(f['Going_Out'].mean()):.2f}")
    st.metric("Avg weekend alcohol (1–5)", f"{float(f['Weekend_Alcohol_Consumption'].mean()):.2f}")
    st.metric("Avg weekday alcohol (1–5)", f"{float(f['Weekday_Alcohol_Consumption'].mean()):.2f}")
    st.metric("Avg health (1–5)", f"{float(f['Health_Status'].mean()):.2f}")


# -------------------------
# Data preview
# -------------------------
st.subheader("👀 Data preview")
st.write("Take a look at the dataset we are exploring.")
with st.expander("Show dataset"):
    st.dataframe(f.head(20))

# -------------------------
# Distributions
# -------------------------
st.divider()
st.subheader("📈 Distributions")
st.write("To understand the data, we must first know the distribution of the features.")

st.write("### Numeric distributions")
num1, num2 = st.columns(2)
with num1:
    st.write("#### Age distribution")
    age_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Age", bin=alt.Bin(maxbins=30), title="Age"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(age_chart, use_container_width=True)

with num2:
    st.write("#### Number of absences distribution")
    abs_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Number_of_Absences", bin=alt.Bin(maxbins=30), title="Number of absences"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(abs_chart, use_container_width=True)

num3, num4 = st.columns(2)
with num3:
    st.write("#### Study time distribution (1–4)")
    study_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Study_Time", bin=alt.Bin(maxbins=4), title="Study time (1–4)"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(study_chart, use_container_width=True)

with num4:
    st.write("#### Health status distribution (1–5)")
    health_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Health_Status", bin=alt.Bin(maxbins=5), title="Health status (1–5)"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(health_chart, use_container_width=True)

st.write("### Categorical distributions")
cat1, cat2 = st.columns(2)
with cat1:
    st.write("#### Gender distribution")
    gender_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Gender", sort="-y", title="Gender"),
            y="count()",
            color=alt.Color("Gender", legend=None),
            tooltip=["Gender", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(gender_chart, use_container_width=True)

with cat2:
    st.write("#### School distribution")
    school_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("School", sort="-y", title="School"),
            y="count()",
            color=alt.Color("School", legend=None),
            tooltip=["School", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(school_chart, use_container_width=True)

cat3, cat4 = st.columns(2)
with cat3:
    st.write("#### Internet access distribution")
    internet_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Internet_Access", sort="-y", title="Internet access"),
            y="count()",
            color=alt.Color("Internet_Access", legend=None),
            tooltip=["Internet_Access", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(internet_chart, use_container_width=True)

with cat4:
    st.write("#### Wants higher education distribution")
    higher_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Wants_Higher_Education", sort="-y", title="Wants higher education"),
            y="count()",
            color=alt.Color("Wants_Higher_Education", legend=None),
            tooltip=["Wants_Higher_Education", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(higher_chart, use_container_width=True)

# -------------------------
# Dropout rate by category
# -------------------------
st.divider()
st.subheader("Dropout rate by category")
st.write("Below we look at the **dropout rate (Dropped_Out)** by some key features.")

cat1, cat2 = st.columns(2)
with cat1:
    st.write("### Dropout rate by school")
    school_stats = f.groupby("School")["Dropped_Out"].mean().reset_index()
    school_chart2 = (
        alt.Chart(school_stats)
        .mark_bar()
        .encode(
            x=alt.X("School", sort="-y", title="School"),
            y=alt.Y("Dropped_Out", title="Dropout rate"),
            color=alt.Color(
                "Dropped_Out",
                scale=alt.Scale(
                    domain=[school_stats["Dropped_Out"].min(), school_stats["Dropped_Out"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["School", "Dropped_Out"]
        )
        .properties(height=350)
    )
    st.altair_chart(school_chart2, use_container_width=True)

with cat2:
    st.write("### Dropout rate by gender")
    gender_stats = f.groupby("Gender")["Dropped_Out"].mean().reset_index()
    gender_chart2 = (
        alt.Chart(gender_stats)
        .mark_bar()
        .encode(
            x=alt.X("Gender", sort="-y", title="Gender"),
            y=alt.Y("Dropped_Out", title="Dropout rate"),
            color=alt.Color(
                "Dropped_Out",
                scale=alt.Scale(
                    domain=[gender_stats["Dropped_Out"].min(), gender_stats["Dropped_Out"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["Gender", "Dropped_Out"]
        )
        .properties(height=350)
    )
    st.altair_chart(gender_chart2, use_container_width=True)

cat3, cat4 = st.columns(2)
with cat3:
    st.write("### Dropout rate by internet access")
    internet_stats = f.groupby("Internet_Access")["Dropped_Out"].mean().reset_index()
    internet_chart2 = (
        alt.Chart(internet_stats)
        .mark_bar()
        .encode(
            x=alt.X("Internet_Access", sort="-y", title="Internet access"),
            y=alt.Y("Dropped_Out", title="Dropout rate"),
            color=alt.Color(
                "Dropped_Out",
                scale=alt.Scale(
                    domain=[internet_stats["Dropped_Out"].min(), internet_stats["Dropped_Out"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["Internet_Access", "Dropped_Out"]
        )
        .properties(height=350)
    )
    st.altair_chart(internet_chart2, use_container_width=True)

with cat4:
    st.write("### Dropout rate by desire for higher education")
    higher_stats = f.groupby("Wants_Higher_Education")["Dropped_Out"].mean().reset_index()
    higher_chart2 = (
        alt.Chart(higher_stats)
        .mark_bar()
        .encode(
            x=alt.X("Wants_Higher_Education", sort="-y", title="Wants higher education"),
            y=alt.Y("Dropped_Out", title="Dropout rate"),
            color=alt.Color(
                "Dropped_Out",
                scale=alt.Scale(
                    domain=[higher_stats["Dropped_Out"].min(), higher_stats["Dropped_Out"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["Wants_Higher_Education", "Dropped_Out"]
        )
        .properties(height=350)
    )
    st.altair_chart(higher_chart2, use_container_width=True)

st.write("### Dropout rate by age")
age_stats = f.groupby("Age")["Dropped_Out"].mean().reset_index()
age_chart2 = (
    alt.Chart(age_stats)
    .mark_bar()
    .encode(
        x=alt.X("Age:O", sort="ascending", title="Age"),
        y=alt.Y("Dropped_Out", title="Dropout rate"),
        color=alt.Color(
            "Dropped_Out",
            scale=alt.Scale(
                domain=[age_stats["Dropped_Out"].min(), age_stats["Dropped_Out"].max()],
                range=["green", "yellow", "red"]
            ),
            legend=None
        ),
        tooltip=["Age", "Dropped_Out"]
    )
    .properties(height=350)
)
st.altair_chart(age_chart2, use_container_width=True)

# -------------------------
# Features vs dropout
# -------------------------
st.divider()
st.subheader("📈 Distributions of the features against dropout")
st.write(
    """In these plots you can see each variable's behaviour with respect to **Dropped_Out**. 
    We strongly recommend to use the outliers filter in *Excluded* mode."""
)

target = "Dropped_Out"
features = [x for x in f.columns if x not in [target]]

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols  # round up
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    # Categorical or low-cardinality features: boxplot
    if f[feature].dtype == "object" or f[feature].nunique() < 10:
        sns.boxplot(x=feature, y=target, data=f, ax=ax)
    else:
        sns.scatterplot(x=feature, y=target, data=f, ax=ax)
    ax.set_title(f"{feature} vs {target}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Remove extra axes if any
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j])

fig.tight_layout()
st.pyplot(fig)

# -------------------------
# Correlation matrix
# -------------------------
st.divider()
st.subheader("Correlation Matrix")
st.write(
    """Here you can see the correlation matrix among all the features (after encoding categorical variables). 
    It is important to focus on the **Dropped_Out** row or column to see its correlation with the features."""
)

fcm = f.copy()

# Encode categorical variables
for col in fcm.columns:
    if fcm[col].dtype == "object":
        le = LabelEncoder()
        fcm[col] = le.fit_transform(fcm[col])
    elif fcm[col].dtype == "bool":
        fcm[col] = fcm[col].astype(int)

corr = fcm.corr().round(2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="icefire", ax=ax)
ax.set_title("Correlation Heatmap", fontsize=10)
st.pyplot(fig)
