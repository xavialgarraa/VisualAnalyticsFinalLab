import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import streamlit.components.v1 as components
import pickle

st.set_page_config(page_title="DriveThru Deals – Model Explainability", page_icon="🚗", layout="wide")
st.title("Model Explainability 🧑‍🏫")
st.write(
    """
    Welcome to the model explainability section. This page helps you understand how the pricing model works. 
    To do so we use the Shapley Additive Explanations (SHAP) values.  
    You can explore:
    - **SHAP introduction:** to understand what are the SHAP values and the test set used for the model.
    - **Global behaviour:** which features matter most overall and how price reacts to mileage, engine volume and year.
    - **Local explanations:** why the model predicted that specific price for your car (or other examples).
    """
)
st.divider()
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)
X_test = pd.read_csv("X_test.csv")
lgb_reg = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

if "lgb_reg" not in globals():
    st.error("The model object `lgb_reg` is not available. Please load it before using this page.")
    st.stop()
if "X_test" not in globals():
    st.error("The dataset `X_test` is not available. Please define or load it before using this page.")
    st.stop()

st.subheader("SHAP Explainer")
st.write(
    """
    **What are SHAP values?**  
    SHAP (SHapley Additive exPlanations) values come from cooperative game theory.
    Each feature is treated like a “player” in a game, and its SHAP value measures
    how much that feature **pushes the prediction up or down** compared to a
    baseline (the model’s average prediction).  
    - Positive SHAP value → the feature makes the predicted price **higher**.  
    - Negative SHAP value → the feature makes the predicted price **lower**.  

    The sum of all SHAP values for a car, plus the baseline, equals the final
      predicted price.
    """
)
with st.spinner("Computing SHAP values for the test set..."):
    shap.initjs()
    explainer = shap.TreeExplainer(lgb_reg)
    shap_values = explainer(X_test)
st.write(
    f"The test set used in this model have **{X_test.shape[0]} cars** and **{X_test.shape[1]} features**."
)
y_pred_all = lgb_reg.predict(X_test)
avg_price_pred = y_pred_all.mean()
st.metric("Average predicted price on test set", f"${avg_price_pred:,.2f}")
st.divider()

# Global feature importance
st.subheader("Global Explainability")
gcol1, gcol2 = st.columns(2)
with gcol1:
    st.write("### SHAP summary plot (feature impact)")
    st.write(
        "Each point is a car in the test set. The color shows the feature value "
        "(red = high, blue = low). The horizontal axis shows how much that feature "
        "pushed the prediction up or down."
    )
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig1)
with gcol2:
    st.write("### SHAP bar plot (mean absolute importance)")
    st.write(
        "This bar chart shows which features are **most important on average** "
        "for the model’s predictions."
    )
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig2)

# Feature-wise SHAP scatter plots
st.subheader("Feature-wise SHAP scatter plots")
st.write(
    "These plots show how the SHAP value (impact on price) changes with the feature value "
    "for the specific variables **mileage**, **engine volume (engV)** and **year**."
)
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    if "mileage" in X_test.columns:
        st.write("#### SHAP vs mileage")
        plt.figure()
        shap.plots.scatter(shap_values[:, "mileage"], show=False)
        fig_m = plt.gcf()
        st.pyplot(fig_m)
        plt.close(fig_m)
    else:
        st.info("Column 'mileage' not found in X_test.")
with fcol2:
    if "engV" in X_test.columns:
        st.write("#### SHAP vs engine volume (engV)")
        plt.figure()
        shap.plots.scatter(shap_values[:, "engV"], show=False)
        fig_e = plt.gcf()
        st.pyplot(fig_e)
        plt.close(fig_e)
    else:
        st.info("Column 'engV' not found in X_test.")
with fcol3:
    if "year" in X_test.columns:
        st.write("#### SHAP vs year")
        plt.figure()
        shap.plots.scatter(shap_values[:, "year"], show=False)
        fig_y = plt.gcf()
        st.pyplot(fig_y)
        plt.close(fig_y)
    else:
        st.info("Column 'year' not found in X_test.")

# Dependence plots with interactions
st.subheader("Dependence plots with interactions")
st.write(
    "Dependence plots reveal **non-linear effects** and interactions. "
    "Here we inspect how **engine volume (engV)** and **year** interact with **mileage**."
)
dcol1, dcol2 = st.columns(2)
with dcol1:
    if {"engV", "mileage"}.issubset(X_test.columns):
        st.write("#### engV vs mileage interaction")
        plt.figure()
        shap.dependence_plot(
            "engV",
            shap_values.values,
            X_test,
            interaction_index="mileage",
            show=False
        )
        fig_dep1 = plt.gcf()
        st.pyplot(fig_dep1)
        plt.close(fig_dep1)
    else:
        st.info("Columns 'engV' and/or 'mileage' not found in X_test.")
with dcol2:
    if {"year", "mileage"}.issubset(X_test.columns):
        st.write("#### year vs mileage interaction")
        plt.figure()
        shap.dependence_plot(
            "year",
            shap_values.values,
            X_test,
            interaction_index="mileage",
            show=False
        )
        fig_dep2 = plt.gcf()
        st.pyplot(fig_dep2)
        plt.close(fig_dep2)
    else:
        st.info("Columns 'year' and/or 'mileage' not found in X_test.")
st.divider()

# Local explanations
st.subheader("Local Explainability")
st.write(
    """
    Choose whether you want to explain the **last car you predicted** in the
    Price Recommender page, or **manually select a car from the test set**.
    """
)
shap.initjs()
explainer = shap.TreeExplainer(lgb_reg)
shap_values_test = explainer(X_test)
mode = st.radio(
    "Which car do you want to explain?",
    (
        "Explain your last predicted car (from Price Recommender)",
        "Pick a car from the test set",
        "Manually enter car features",
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
if mode == "Explain your last predicted car (from Price Recommender)":
    if has_last_pred:
        X_explain = st.session_state["explain_X_encoded"]
        X_explain_display = st.session_state["explain_X_display"]
        y_explain_pred = st.session_state["explain_y_pred"]
        shap_values_explain = explainer(X_explain)
        shap_row = shap_values_explain[0]
        st.info("You are explaining the **last car you predicted** in the Price Recommender page.")
    else:
        st.warning(
            "No previous prediction found from the Price Recommender page. "
            "Please predict a car there first, meanwhile you can choose a car from the test set.")
        use_test_set = True
elif mode == "Pick a car from the test set":
    use_test_set = True
if use_test_set and mode != "Manually enter car features":
    instance_index = st.slider("Select a test instance index",min_value=0,max_value=X_test.shape[0] - 1,value=0,step=1,help="Choose which car from the test set you want to inspect.",)
    X_explain = X_test.iloc[[instance_index]]
    X_explain_display = X_explain.copy()
    try:
        X_explain_display["car"] = le_car.inverse_transform(X_explain_display["car"].astype(int))
        X_explain_display["body"] = le_body.inverse_transform(X_explain_display["body"].astype(int))
        X_explain_display["engType"] = le_engType.inverse_transform(X_explain_display["engType"].astype(int))
        X_explain_display["drive"] = le_drive.inverse_transform(X_explain_display["drive"].astype(int))
    except Exception as e:
        st.warning(f"Could not inverse-transform some categorical features: {e}")
    X_explain_display["registration"] = X_explain_display["registration"].map({1: "yes", 0: "no"})
    y_explain_pred = float(lgb_reg.predict(X_explain)[0])
    shap_row = shap_values_test[instance_index]
    st.info(f"You are explaining **car #{instance_index} from the test set**.")
if mode == "Manually enter car features":
    st.markdown("#### Enter the car features you want to explain")
    mcol1, mcol2, mcol3, mcol4= st.columns(4)
    with mcol1:
        man_car = st.selectbox("Car Brand", le_car.classes_, key="exp_manual_car",help="Select the brand of the car. If there is not your car's brand select Other.")
        man_registration = st.selectbox("Registered", ["yes", "no"], key="exp_manual_registration",help="Indicate whether the car is officially registered.")
    with mcol2:
        man_body = st.selectbox("Body Type", le_body.classes_, key="exp_manual_body",help="Choose the car's body configuration. If there is not your car's body configuration select Other.")
        man_year = st.slider("Year",min_value=1950,max_value=2025,step=1,help="Select the manufacturing year of the car.")
    with mcol3:
        man_drive = st.selectbox("Drive type", le_drive.classes_, key="exp_manual_drive",help="Choose the drivetrain: front-wheel, rear-wheel, or full/all-wheel drive.")
        man_mileage = st.slider("Mileage (km)",min_value=0,max_value=1000,step=10,help="Mileage is entered in thousands. Example: 10 = 10,000 km.")
    with mcol4:
        man_engType = st.selectbox("Engine Type", le_engType.classes_, key="exp_manual_engType",help="Select the fuel or engine type of the vehicle.")
        man_engV = st.slider("Engine Volume (L)",min_value=0.0,max_value=30.0,step=0.1,help="Select the engine displacement in litres.")
    manual_display = pd.DataFrame(
        [[
            man_car,
            man_body,
            man_mileage,
            man_engV,
            man_engType,
            man_registration,
            man_year,
            man_drive,
        ]],
        columns=['car', 'body', 'mileage', 'engV', 'engType', 'registration', 'year', 'drive'],
    )
    manual_encoded = manual_display.copy()
    manual_encoded["car"] = le_car.transform(manual_encoded["car"])
    manual_encoded["body"] = le_body.transform(manual_encoded["body"])
    manual_encoded["engType"] = le_engType.transform(manual_encoded["engType"])
    manual_encoded["drive"] = le_drive.transform(manual_encoded["drive"])
    manual_encoded["registration"] = manual_encoded["registration"].map({"yes": 1, "no": 0})
    X_explain = manual_encoded
    X_explain_display = manual_display
    y_explain_pred = float(lgb_reg.predict(X_explain)[0])
    shap_row = explainer(X_explain)[0]
    st.info("You are explaining a **custom car** defined by the inputs above.")
if X_explain_display is not None and y_explain_pred is not None:
    st.write("### Selected car features")
    st.dataframe(X_explain_display)
    st.metric("Predicted price for this car", f"{y_explain_pred:,.2f} €")
else:
    st.stop()

st.subheader("SHAP Waterfall plot")
st.write(
    """
    The **waterfall plot** shows how each feature contributes to move the prediction
    from the model's **base value** (average prediction) to the final **price for this car**.
    """
)
with st.spinner("Rendering waterfall plot..."):
    plt.figure()
    shap.plots.waterfall(shap_row, show=False)
    fig_w = plt.gcf()
    st.pyplot(fig_w)
    plt.close(fig_w)

st.subheader("SHAP Force plot")
st.write(
    """
    The **force plot** provides a compact view of how features push the prediction
    above (red) or below (blue) the base value for this specific car.
    """
)
with st.spinner("Rendering force plot..."):
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_row.values,
        X_explain.iloc[0, :],
        matplotlib=False
    )
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(force_html, height=220, scrolling=True)

st.subheader("SHAP Decision plot")
st.write(
    """
    The **decision plot** shows how the model gradually builds the prediction by adding
    feature contributions one by one, from the base value to the final price of this car.
    """
)
with st.spinner("Rendering decision plot..."):
    plt.figure()
    shap.decision_plot(
        base_value=shap_row.base_values,
        shap_values=shap_row.values,
        features=X_explain.iloc[0, :],
        show=False
    )
    fig_dec = plt.gcf()
    st.pyplot(fig_dec)
    plt.close(fig_dec)
