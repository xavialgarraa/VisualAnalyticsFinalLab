import streamlit as st

st.set_page_config(
    page_title="DriveThru Deals",
    page_icon="🚗",
    layout="wide"
)

st.title("DriveThru Deals 🤝🚗")
st.write(
    """Welcome to the best car buying and selling company!  
    Explore the market, get a fair price suggestion, and understand why."""
)
st.divider()
st.subheader("What is DriveThru Deals?")
st.write(
    """
    DriveThru Deals is the a car buying and selling platform that helps you answer a key question:  
    “Given the characteristics of my car, what is a fair price to list it for?”   
    
    In this app you will use a real dataset of more than 8,800 car listings to:  
    - Inspect how prices vary across brands, body types, mileage, year, and more  
    - Get a price recommendation for a specific car  
    - Explore explainability tools that show which features drive the suggested price  
    
    The goal is not only to predict a number, but to build trust by making the model’s
    reasoning clear and understandable.
    """
)

st.divider()


st.subheader("How this app is organised")

c1, c2, c3 = st.columns(3)

with c1:
    st.write("")
    st.write("### 1. Data Explorer")
    st.write(
        """
        Use this section to get a feeling for the market:

        • Summary statistics and distributions  
        • Brand, body type, and drive type comparisons  
        • How mileage, engine volume and year affect price  
        """
    )

with c2:
    st.write("")
    st.write("### 2. Price Recommender")
    st.write(
        """
        Enter the characteristics of a car:

        • Brand, model and body type  
        • Year, mileage, engine volume and type  
        • Drive type and registration status  

        The model will output a suggested listing price based on similar cars.
        """
    )

with c3:
    st.write("")
    st.write("### 3. Predictor Explainability")
    st.write(
        """
        This section focuses on “why this prediction”:

        • Global feature importance  
        • Local explanations for a specific car  
        • Visualisations showing what pushes price up or down  
        """
    )

st.divider()

st.subheader("🚦 Quick start 🚦")

st.write(
    """
    1. Start with the Data Explorer  
       Look at basic plots and patterns.

    2. Move to the Price Recommender  
       Enter car information and get your predicted price.

    3. Go to the Explainability section  
       Understand why the model recommended that price.

    4. Experiment  
       Change mileage, year or other features and observe how the suggested price will change in the future 
       (or even look for the price of your vehicle in the past).
    """
)
