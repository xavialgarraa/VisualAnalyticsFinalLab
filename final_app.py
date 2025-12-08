import streamlit as st

st.set_page_config(
    page_title="Student Dropout – Family Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("Understand and Reduce Your Child’s Dropout Risk 🎓👨‍👩‍👧")

st.write(
    """
    Welcome to the **Student Dropout – Family Assistant**.  

    This application is designed to help **families, tutors and guardians** better understand  
    the risk that a student could **leave school early (dropout)** and what factors are 
    most related to that risk.

    Using information from a real dataset of students with similar characteristics, the app lets you:

    - See how a set of students behave (absences, study time, support, etc.).
    - Estimate your child’s **dropout risk (in percentage)** from their personal, family and academic context.
    - Explore **“what-if” changes** (more study time, fewer absences, extra support…) and see how much these changes could help.
    - Understand **why** the model gives a certain risk, with visual explanations that are easier to interpret.
    """
)

st.divider()

# HOW THE APP IS ORGANISED
st.subheader("How this app is organised")

c1, c2 = st.columns(2)
c11, c22 = st.columns(2)
with c1:
    st.write("### 1. Data Exploration 🔎")
    st.write(
        """
        Explore students **similar to your child** using filters  
        (age, study time, absences, support, school, etc.).

        View:
        - Dropout rate of similar students  
        - Key statistics (absences, failures, study time…)  
        - How dropout varies with habits and support  

        Ideal for understanding the **student’s context**.
        """
    )

with c2:
    st.write("### 3. Changes Impact Simulator ⚙️")
    st.write(
        """
        Simulate **multiple changes** (personal, family, school) at once.  

        See which actions **reduce risk the most** and explore helpful suggestions  
        like support programmes or tutoring.
        """
    )
with c11:
    st.write("### 2. Dropout Predictor 🎯")
    st.write(
        """
        Enter the student’s **personal, family and school information** to:
        - Predict dropout risk (%)  
        - Get a risk category (LOW / MEDIUM / HIGH)  

        Use **What-If buttons** to test improvements  
        (more study time, fewer absences, more support…).
        """
    )

with c22:
    st.write("### 4. Model Explainability 🧠")
    st.write(
        """
        Understand **why** the model made a prediction using  
        SHAP visual explanations.

        Useful for discussions with teachers or support teams.  
        Works with:
        - Your last predicted student  
        - A student from the dataset  
        - A custom student you define
        """
    )

st.divider()

# QUICK START FOR FAMILIES
st.subheader("🚦 Quick start guide")

st.markdown(
    """
    1. **Start with the Data Exploration page**  
       - Use the filters to focus on students similar to your child (same age, school, support…).  
       - Look at the **dropout rate** and the typical profile of this group (absences, failures, support…).

    2. **Go to the Dropout Predictor**  
       - Enter the student’s personal, family and academic information.  
       - Check the **predicted dropout risk (in %)** and whether it is LOW, MEDIUM or HIGH.  
       - Use the **What-If buttons** to see which changes (more study time, fewer absences, extra support…)  
         would reduce the risk the most.

    3. **Open the Changes Impact Simulator**  
       - Use this page if you want to **compare many possible actions** at once.  
       - Explore **personal**, **family** and **school** changes and see which ones have  
         the **strongest impact** (positive or negative) on the student’s dropout risk.  
       - Review the **Scholarships** and **Recommendations** boxes to get concrete ideas for support.

    4. **Use the Model Explainability page for deeper understanding**  
       - If you need to explain the prediction to someone else (another parent, a teacher,  
         a psychologist or social worker), this page shows **why** the model made that prediction.  
       - Look at which variables are helping to **protect** the student and which ones are  
         pushing the risk **higher**.

    5. **Talk together and plan realistic actions**  
       - This app is not a substitute for teachers or mental health professionals,  
         but it can start a **constructive conversation** between families, tutors and the student.  
       - Use the insights to decide **which small, realistic changes** could help keep the student  
         **engaged in school** and reduce the chance of dropout.
    """
)