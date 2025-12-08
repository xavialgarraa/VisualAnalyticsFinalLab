import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from math import pi

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Student Dropout – Ultimate Explorer",
    page_icon="🎓",
    layout="wide",
)

# -------------------------
# Load and preprocess data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("student_dropout.csv")
    
    # Drop grade columns (Focus on structural causes)
    cols_to_drop = ["Grade_1", "Grade_2", "Final_Grade"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df.dropna()
    
    # Create Integer Target for calculations (Mean = Dropout Rate)
    df['Dropped_Out_Int'] = df['Dropped_Out'].astype(int)
    
    return df

df = load_data()

# -------------------------
# Sidebar: Advanced Filters
# -------------------------
st.sidebar.header("🎛️ Filter Dataset")
st.sidebar.write("Refine your student cohort below:")

# Helper functions for sidebar
def radio_three(label, options, key, help_text=""):
    options = sorted(list(options))
    selected = st.sidebar.radio(label, ["All"] + options, key=key, help=help_text)
    return options if selected == "All" else [selected]

def radio_yes_no_both(label, column_name, key_prefix):
    values = sorted(df[column_name].unique())
    choice = st.sidebar.radio(label, ["All"] + list(values), index=0, key=f"{key_prefix}_radio")
    return values if choice == "All" else [choice]

# --- 1. Demographics ---
with st.sidebar.expander("👤 Demographics", expanded=True):
    sel_school = radio_three("School", df["School"].unique(), "school_f")
    sel_gender = radio_three("Gender", df["Gender"].unique(), "gender_f")
    sel_address = radio_three("Address (Urban/Rural)", df["Address"].unique(), "address_f")
    sel_famsize = radio_three("Family Size", df["Family_Size"].unique(), "famsize_f")
    sel_pstatus = radio_three("Parental Status", df["Parental_Status"].unique(), "pstatus_f")
    
    age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

# --- 2. Academic & Support ---
with st.sidebar.expander("📚 Academic & Support", expanded=False):
    sel_schoolsup = radio_yes_no_both("School Support", "School_Support", "schoolsup")
    sel_famsup = radio_yes_no_both("Family Support", "Family_Support", "famsup")
    sel_paid = radio_yes_no_both("Extra Paid Class", "Extra_Paid_Class", "paid")
    sel_higher = radio_yes_no_both("Wants Higher Edu", "Wants_Higher_Education", "higher")
    sel_internet = radio_yes_no_both("Internet Access", "Internet_Access", "internet")
    
    study_range = st.sidebar.slider("Weekly Study Time (1-4)", int(df["Study_Time"].min()), int(df["Study_Time"].max()), (1, 4))
    absences_range = st.sidebar.slider("Absences", int(df["Number_of_Absences"].min()), int(df["Number_of_Absences"].max()), (0, 93))
    failures_range = st.sidebar.slider("Past Failures", int(df["Number_of_Failures"].min()), int(df["Number_of_Failures"].max()), (0, 3))

# --- 3. Lifestyle & Family ---
with st.sidebar.expander("🏠 Lifestyle & Family", expanded=False):
    sel_romantic = radio_yes_no_both("In Relationship", "In_Relationship", "romantic")
    sel_goout = st.sidebar.slider("Going Out (1-5)", 1, 5, (1, 5))
    sel_walc = st.sidebar.slider("Weekend Alcohol (1-5)", 1, 5, (1, 5))
    sel_health = st.sidebar.slider("Health Status (1-5)", 1, 5, (1, 5))

# --- Apply Filters ---
mask = (
    df["School"].isin(sel_school) &
    df["Gender"].isin(sel_gender) &
    df["Address"].isin(sel_address) &
    df["Family_Size"].isin(sel_famsize) &
    df["Parental_Status"].isin(sel_pstatus) &
    df["School_Support"].isin(sel_schoolsup) &
    df["Family_Support"].isin(sel_famsup) &
    df["Extra_Paid_Class"].isin(sel_paid) &
    df["Wants_Higher_Education"].isin(sel_higher) &
    df["Internet_Access"].isin(sel_internet) &
    df["In_Relationship"].isin(sel_romantic) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Study_Time"].between(study_range[0], study_range[1])) &
    (df["Number_of_Absences"].between(absences_range[0], absences_range[1])) &
    (df["Number_of_Failures"].between(failures_range[0], failures_range[1])) &
    (df["Going_Out"].between(sel_goout[0], sel_goout[1])) &
    (df["Weekend_Alcohol_Consumption"].between(sel_walc[0], sel_walc[1])) &
    (df["Health_Status"].between(sel_health[0], sel_health[1]))
)

f = df[mask]

if len(f) == 0:
    st.error("⚠️ No students match the selected filters. Please relax your filters in the sidebar.")
    st.stop()

# -------------------------
# Main Dashboard
# -------------------------

st.title("🎓 Student Dropout Analytics Dashboard")
st.markdown(f"""
**Target Population:** {len(f)} students  
**Current Dropout Rate:** <span style='color:red; font-size:1.2em; font-weight:bold'>{f['Dropped_Out_Int'].mean()*100:.1f}%</span>
""", unsafe_allow_html=True)

# KPI Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Students", len(f))
kpi2.metric("Dropouts", f[f['Dropped_Out_Int']==1].shape[0])
kpi3.metric("Avg Age", f"{f['Age'].mean():.1f}")
kpi4.metric("Avg Absences", f"{f['Number_of_Absences'].mean():.1f}")
kpi5.metric("Avg Failures", f"{f['Number_of_Failures'].mean():.2f}")

st.divider()

# -------------------------
# TABS STRUCTURE
# -------------------------
# REORDERED TABS AS REQUESTED
tab_data, tab_explore, tab_corr, tab_deep, tab_profile = st.tabs([
    "📂 Data & Distributions", 
    "📊 Features vs Target",
    "🔗 Correlation Matrix",
    "🔍 Strategic Insights",  
    "🧠 Behavioral Profile (Radar)"
])

# =========================================
# TAB 1: Data & Distributions (NEW)
# =========================================
with tab_data:
    st.header("Dataset Overview")
    st.write("Review the raw data and the general distribution of variables (Univariate Analysis).")
    
    # 1. Show Dataframe
    with st.expander("👀 View Raw Dataframe", expanded=False):
        st.dataframe(f, use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(f.describe(), use_container_width=True)
    
    st.divider()

    # Data Dictionary
    with st.expander("📖 Open Data Dictionary (Column Definitions)", expanded=False):
        st.markdown("""
        | Column | Description |
        | :--- | :--- |
        | **School** | Name of the school (e.g., MS, GP). |
        | **Age** | Age of the student. |
        | **Address** | Residence type (U: Urban, R: Rural). |
        | **Mother_Education/Father_Education** | 0 (None) to 4 (Higher Ed). |
        | **Mother_Job/Father_Job** | Job type (teacher, health, services, etc). |
        | **Study_Time** | Weekly study hours (1: <2h, 4: >10h). |
        | **Number_of_Failures** | Number of past class failures. |
        | **School_Support** | Extra educational support (yes/no). |
        | **Internet_Access** | Internet availability at home. |
        | **Family_Relationship** | Quality of family relationships (1: Bad - 5: Excellent). |
        | **Free_Time** | Free time after school (1-5). |
        | **Going_Out** | Frequency of going out with friends (1-5). |
        | **Weekend_Alcohol_Consumption** | Consumption level (1-5). |
        | **Health_Status** | Current health (1: Very Bad - 5: Excellent). |
        | **Number_of_Absences** | Total school absences. |
        | **Dropped_Out** | Target variable (True/False). |
        """)

    st.divider()


    # 2. Variable Distributions
    st.subheader("General Variable Distributions")
    st.write("These plots show the counts of each variable in the current filtered selection.")
    
    # Split columns
    ignored_cols = ["Dropped_Out_Int", "Dropped_Out"]
    num_vars = [c for c in f.select_dtypes(include=np.number).columns if c not in ignored_cols]
    cat_vars = [c for c in f.select_dtypes(include=['object']).columns if c not in ignored_cols]
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("### 🔢 Numeric Variables")
        # Histogram for numeric
        target_num_col = st.selectbox("Select Numeric Variable", num_vars, index=0)
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        sns.histplot(f[target_num_col], kde=True, ax=ax_hist, color="skyblue")
        ax_hist.set_title(f"Distribution of {target_num_col}")
        st.pyplot(fig_hist)
        
    with col_d2:
        st.markdown("### 🔤 Categorical Variables")
        # Countplot for categorical
        target_cat_col = st.selectbox("Select Categorical Variable", cat_vars, index=0)
        fig_count, ax_count = plt.subplots(figsize=(6, 4))
        sns.countplot(y=f[target_cat_col], ax=ax_count, palette="viridis", order=f[target_cat_col].value_counts().index)
        ax_count.set_title(f"Count of {target_cat_col}")
        st.pyplot(fig_count)
        
    # Quick Grid View option
    with st.expander("Show all small distributions (Grid View)"):
        st.write("Small multiples for quick scanning.")
        # Plot top 6 categoricals
        fig_grid, axes_grid = plt.subplots(4, 3, figsize=(12, 12))
        axes_grid = axes_grid.flatten()
        for i, col in enumerate(cat_vars[:12]):
            sns.countplot(x=col, data=f, ax=axes_grid[i], palette="pastel")
            axes_grid[i].tick_params(axis='x', rotation=45, labelsize=8)
            axes_grid[i].set_xlabel("")
            axes_grid[i].set_title(col, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_grid)
# =========================================
# TAB 2: Feature Explorer (General)
# =========================================
with tab_explore:
    st.header("Feature Analysis (vs Target)")
    st.write("Explore how specific variables relate to **Dropout**.")
    
    # Identify column types
    target = "Dropped_Out"
    ignored_cols = ["Dropped_Out_Int", "Dropped_Out"]
    num_vars = [c for c in f.select_dtypes(include=np.number).columns if c not in ignored_cols]
    cat_vars = [c for c in f.select_dtypes(include=['object']).columns if c not in ignored_cols]

    # --- NUMERIC ---
    st.subheader("Numeric Distributions (Boxplots)")
    selected_num = st.multiselect("Choose numeric variables:", num_vars, default=["Age", "Number_of_Absences", "Study_Time"])
    if selected_num:
        cols = st.columns(len(selected_num))
        for i, col in enumerate(selected_num):
            with cols[i]:
                fig_n, ax_n = plt.subplots(figsize=(4, 5))
                sns.boxplot(x=target, y=col, data=f, palette="Set2", ax=ax_n)
                ax_n.set_title(col)
                st.pyplot(fig_n)

    st.divider()

    # --- CATEGORICAL ---
    st.subheader("Categorical Breakdown (Stacked Bar)")
    st.caption("View the proportion of Dropouts within each category.")
    selected_cat = st.multiselect("Choose categorical variables:", cat_vars, default=["Mother_Job", "Reason_for_Choosing_School"])
    
    if selected_cat:
        for col in selected_cat:
            # Create a cross tab normalized by index to get percentages
            ct = pd.crosstab(f[col], f['Dropped_Out'], normalize='index') * 100
            
            st.write(f"**{col}**")
            # Stacked bar chart
            fig_stack, ax_stack = plt.subplots(figsize=(10, 4))
            ct.plot(kind='barh', stacked=True, color=['#87CEFA', '#FA8072'], ax=ax_stack) # Blue for False, Red for True
            
            ax_stack.set_xlabel("Percentage (%)")
            ax_stack.set_ylabel(col)
            ax_stack.legend(title="Dropped Out", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Annotate
            for c in ax_stack.containers:
                ax_stack.bar_label(c, fmt='%.0f%%', label_type='center', color='black')
                
            st.pyplot(fig_stack)

# =========================================
# TAB 4: Correlation Matrix (Improved)
# =========================================

with tab_corr:

    st.header("Correlation Analysis")
    st.write("What relates most strongly with dropout?")

    # --- Encoding ---
    f_enc = f.copy()
    for c in f_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        f_enc[c] = le.fit_transform(f_enc[c])

    if 'Dropped_Out' in f_enc.columns:
        f_enc = f_enc.drop(columns=['Dropped_Out'])

    # Raw correlation
    corr = f_enc.corr()

    # ---- Controls ----

    # Threshold
    thresh = st.slider(
        "Correlation Threshold (abs value)",
        0.0, 0.5, 0.10, 0.01
    )

    # Variable types
    var_types = st.multiselect(
        "Select variable types to include",
        ["numeric", "categorical"],
        default=["numeric", "categorical"]
    )

    # Determine columns from variable type
    use_cols = []
    if "numeric" in var_types:
        use_cols += f_enc.select_dtypes(include='number').columns.tolist()
    if "categorical" in var_types:
        use_cols += f_enc.select_dtypes(exclude='number').columns.tolist()

    corr = f_enc[use_cols].corr()

    # Compute target correlations and apply threshold
    target_c = corr['Dropped_Out_Int'].abs().sort_values(ascending=False)
    top_feats = target_c[target_c > thresh].index.tolist()

    # Let user manually add/remove vars
    vars_user = st.multiselect(
        "Choose variables to display",
        options=use_cols,
        default=top_feats
    )

    # Choose color palette
    palette = st.selectbox(
        "Color palette",
        ["coolwarm", "viridis", "magma", "crest", "icefire"],
        index=0
    )

    # ---- Output information ----

    st.subheader("Top Variables (sorted by correlation)")
    st.dataframe(target_c.to_frame("Abs Corr w/ Dropout").head(15))

    # ---- Heatmap ----
    if len(vars_user) >= 2:
        fig_c, ax_c = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr.loc[vars_user, vars_user],
            annot=True,
            cmap=palette,
            fmt=".2f",
            center=0,
            ax=ax_c
        )
        st.pyplot(fig_c)
    else:
        st.warning("Please select at least two variables to build the matrix.")

# =========================================
# TAB 4: Strategic Insights 
# =========================================
with tab_deep:
    st.header("🚀 Strategic Deep Dive: The 'Why' Behind Dropout")
    st.markdown("""
    Based on the correlation analysis, we have identified the **three strongest drivers** of student dropout. 
    This section visualizes how these critical factors interact to increase risk.
    """)

    # --- SECTION 1: THE ACADEMIC SPIRAL (Failures) ---
    st.subheader("1. The 'Failure Trap' (Correlation: +0.38)")
    st.write("The strongest predictor of dropout is the number of past class failures. The risk is not linear; it compounds exponentially.")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Combined Line & Bar Chart for Impact
        fail_data = f.groupby('Number_of_Failures')['Dropped_Out_Int'].agg(['mean', 'count']).reset_index()
        fail_data.columns = ['Failures', 'Dropout_Rate', 'Student_Count']
        
        fig_fail, ax1 = plt.subplots(figsize=(8, 4))
        
        # Bar chart for Volume
        sns.barplot(x='Failures', y='Student_Count', data=fail_data, alpha=0.3, color='gray', ax=ax1)
        ax1.set_ylabel("Number of Students (Volume)", color='gray')
        
        # Line chart for Risk
        ax2 = ax1.twinx()
        sns.lineplot(x='Failures', y='Dropout_Rate', data=fail_data, marker='o', color='#D9534F', linewidth=3, ax=ax2)
        ax2.set_ylabel("Dropout Probability", color='#D9534F')
        ax2.set_ylim(0, 1.1)
        
        # Annotate
        for i, txt in enumerate(fail_data['Dropout_Rate']):
            ax2.text(i, txt + 0.05, f"{txt*100:.0f}%", ha='center', color='#D9534F', fontweight='bold')
            
        ax1.set_title("Dropout Rate skyrockets after just 1 failure")
        st.pyplot(fig_fail)
        
    with c2:
        st.warning(
            """
            **🚨 Critical Insight:**
            
            Students with **0 failures** are generally safe.
            
            However, notice the jump at **1 Failure**. 
            This suggests that early intervention immediately after the *first* academic stumble is the most effective way to prevent dropout. 
            Waiting for a second failure is often too late.
            """
        )

    st.divider()

    # --- SECTION 2: INSTITUTIONAL & GEOGRAPHIC FACTORS (School & Address) ---
    st.subheader("2. Institutional & Environmental Context (Correlation: +0.30)")
    st.write("The data shows a significant disparity between schools and residence types. Is the environment creating barriers?")

    c3, c4 = st.columns(2)
    
    with c3:
        # Interaction between School and Address
        st.markdown("**Dropout Rate by School & Residence**")
        fig_env, ax_env = plt.subplots()
        
        # Calculate means
        env_pivot = f.groupby(['School', 'Address'])['Dropped_Out_Int'].mean().reset_index()
        
        sns.barplot(x='School', y='Dropped_Out_Int', hue='Address', data=env_pivot, palette="viridis", ax=ax_env)
        ax_env.set_ylabel("Dropout Probability")
        ax_env.set_ylim(0, 1)
        
        # Add labels
        for container in ax_env.containers:
            ax_env.bar_label(container, fmt='%.2f')
            
        st.pyplot(fig_env)
        st.caption("U = Urban, R = Rural.")

    with c4:
        st.info(
            """
            **🏫 Institutional Barrier Insight**
            Students from the MS school display significantly higher dropout rates, regardless 
            of whether they live in urban or rural areas. This suggests that the school environment 
            itself may be a critical driver of dropout.
            """
        )

        if 'Travel_Time' in f.columns:
            avg_travel = f.groupby('Dropped_Out')['Travel_Time'].mean()
            diff = avg_travel[True] - avg_travel[False]
            if diff > 0:
                st.markdown(
                    f"""
                    ⏳ **Travel Burden:** Students who drop out spend **{diff:.2f} more units of time** traveling on average.
                    Longer commutes may reduce school engagement and attendance.
                    """
                )

    st.divider()

    # --- SECTION 3: MOTIVATION & PARENTAL SUPPORT ---
    st.subheader("3. The 'Ambition Buffer' (Correlation: -0.31)")
    st.write("Desire for higher education is the strongest *negative* correlation, meaning it acts as a shield against dropout.")

    c5, c6 = st.columns([1, 2])

    with c5:
        st.markdown("**Impact of Higher Ed Goals**")
        # Simple donut chart or bar
        goal_rate = f.groupby('Wants_Higher_Education')['Dropped_Out_Int'].mean() * 100
        
        fig_goal, ax_goal = plt.subplots(figsize=(4, 4))
        colors = ['#ff9999','#66b3ff'] # Red for No desire, Blue for Yes desire
        # Use simple bar for clarity
        sns.barplot(x=goal_rate.index, y=goal_rate.values, palette=["#D9534F", "#5BC0DE"], ax=ax_goal)
        ax_goal.set_ylabel("Dropout Rate (%)")
        ax_goal.set_xlabel("Wants Higher Edu?")
        for container in ax_goal.containers:
            ax_goal.bar_label(container, fmt='%.1f%%')
        st.pyplot(fig_goal)

    with c6:
        st.markdown("**Does Study Time Strengthen the Ambition Effect?**")

        fig_complex, ax_complex = plt.subplots(figsize=(8, 4))

        # Interaction: Study Time vs Ambition vs Dropout
        sns.lineplot(
            data=f,
            x='Study_Time',
            y='Dropped_Out_Int',
            hue='Wants_Higher_Education',
            marker="o",
            palette={'no': "red", 'yes': "blue"},
            ax=ax_complex
        )

        ax_complex.set_title("Dropout Risk: Study Time (X) vs Student Ambition (Color)")
        ax_complex.set_ylabel("Dropout Probability")
        ax_complex.set_xlabel("Weekly Study Time (1–4)")
        st.pyplot(fig_complex)

        st.success(
            """
            **🧠 The Insight:**
            Even when students study the same amount, **those who want higher education consistently
            show lower dropout risk** (Blue) than those who don't (Red).
            
            **Actionable Strategy:** Motivation-oriented mentoring + structured study habits is a powerful
            combined intervention.
            """
        )


# =========================================
# TAB 5: Behavioral Profile (Radar Chart) 
# =========================================
with tab_profile:
    st.header("The 'Persona' Comparison")
    st.write("Compare the average habits and personality traits of students who drop out vs. those who stay.")

    col_radar, col_info = st.columns([2, 1])

    with col_radar:
        # Prepare Data for Radar Plot
        features_radar = ['Family_Relationship', 'Free_Time', 'Going_Out', 
                          'Weekend_Alcohol_Consumption', 'Weekday_Alcohol_Consumption', 'Health_Status']
        
        # Calculate means for both groups
        radar_df = f.groupby('Dropped_Out')[features_radar].mean().reset_index()
        
        # Function to draw radar chart
        def make_radar(df_radar, categories):
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1] # Close the loop
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            
            # Draw one axe per variable + labels
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
            plt.ylim(0, 5)
            
            # Plot data
            # Group 0: No Dropout (Blue)
            if False in df_radar['Dropped_Out'].values:
                values0 = df_radar[df_radar['Dropped_Out']==False][categories].values.flatten().tolist()
                values0 += values0[:1]
                ax.plot(angles, values0, linewidth=1, linestyle='solid', label="Stays (False)", color="blue")
                ax.fill(angles, values0, 'blue', alpha=0.1)
            
            # Group 1: Dropout (Red)
            if True in df_radar['Dropped_Out'].values:
                values1 = df_radar[df_radar['Dropped_Out']==True][categories].values.flatten().tolist()
                values1 += values1[:1]
                ax.plot(angles, values1, linewidth=1, linestyle='solid', label="Drops Out (True)", color="red")
                ax.fill(angles, values1, 'red', alpha=0.1)
                
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            return fig

        if len(radar_df) > 0:
            st.pyplot(make_radar(radar_df, features_radar))
        else:
            st.warning("Not enough data to generate radar chart.")

    with col_info:
        st.info(
            """
            **How to read this chart:**
            - **Center (0):** Very Low
            - **Outer Edge (5):** Very High
            
            **Patterns to look for:**
            - Does the **Red shape** stretch more towards *Alcohol* or *Going Out*?
            - Does the **Blue shape** stretch more towards *Family Relationship* or *Health*?
            """
        )