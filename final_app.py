import streamlit as st

# -------------------------
# CONFIG STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Student Dropout – Family Assistant",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# LANDING PAGE
# -------------------------
st.title("Understand Your Child’s Dropout Risk 🎓👨‍👩‍👧")

st.write(
    """
    Welcome to the **Student Dropout – Family Assistant**.  

    This application is designed to help **families, tutors and guardians** better understand  
    the risk that a student could **leave school early (dropout)** and what factors are 
    most related to that risk.

    Using a real dataset of students with similar characteristics, the app allows you to:

    - Explore how students similar to your child behave (absences, study time, support, etc.).
    - Estimate your child’s **dropout risk (in percentage)** from their personal, family and academic context.
    - See **which factors are helping or increasing this risk**, in a visual and understandable way.
    """
)

st.divider()

st.subheader("How this app is organised for families")

c1, c2, c3 = st.columns(3)

with c1:
    st.write("### 1. Data Exploration")
    st.write(
        """
        Use this page if you want to understand the **general context**:

        - Filter the data to focus on students similar to your child  
          (same school, age range, gender, internet access, etc.).  
        - See summary metrics such as average age, dropout rate, absences, study time…  
        - Check how **dropout** changes depending on **internet access, desire for higher education, age**, etc.  
        - Look at correlations to see which factors tend to move together with dropout.
        
        This helps you answer questions like:  
        *“Among students similar to my child, how common is dropout?”*
        """
    )

with c2:
    st.write("### 2. Dropout Predictor")
    st.write(
        """
        Here you can focus on **one specific student**, for example **your son or daughter**:

        - Enter the student’s **personal**, **family** and **school** information  
          (age, study time, absences, support, etc.).  
        - Obtain a **dropout risk prediction (in %)** and a **risk level** (low / medium / high).  
        - Use the **What-If buttons** to see how the risk would change if:
          - the student studied more or less,  
          - had fewer absences,  
          - received more school support,  
          - or changed some habits (going out, alcohol, free time…).  

        This can help families think about **realistic changes** that may reduce the risk.
        """
    )

with c3:
    st.write("### 3. Model Explainability")
    st.write(
        """
        This page explains **why** the model gives a certain dropout risk.

        - You’ll see a short explanation of **SHAP values**, a method that shows  
          how each factor (study time, absences, support, etc.) **pushes the risk up or down**.  
        - You can explore the **overall behaviour** of the model:  
          which features are usually more important across many students.  
        - You can also obtain **local explanations** for:
          - The last student you analysed in the Dropout Predictor,  
          - A student from the dataset, or  
          - A **new custom student** that you enter manually.  

        Visual plots (waterfall, force, decision plots) show how each characteristic helps  
        **protect** your child or **increase** their risk of dropout.
        """
    )

st.divider()

st.subheader("🚦 Quick start guide for parents and guardians")

st.markdown(
    """
    1. **Start with the Data Exploration page**  
       Choose filters (age, school, support, etc.) to see **students similar to your child**.  
       Look at the dropout rate and main characteristics of this group.

    2. **Go to the Dropout Predictor**  
       Enter the student’s information as accurately as possible.  
       Check the **dropout risk (in %)** and the risk level (low / medium / high).

    3. **Use the What-If analysis**  
       Try changing one thing at a time: more study time, fewer absences, more support,  
       less alcohol, etc.  
       See **which changes would reduce the risk** and by how much.

    4. **Open the Model Explainability page**  
       If you want to understand in more detail *why* the model gave that risk,  
       use the SHAP explanations to see which factors are **most influential** for that student.

    5. **Use the insights to talk and plan**  
       The app is not a replacement for teachers or psychologists,  
       but it can start a **conversation** between families, tutors and the student  
       about what could be improved to **keep the student engaged in school**.
    """
)

