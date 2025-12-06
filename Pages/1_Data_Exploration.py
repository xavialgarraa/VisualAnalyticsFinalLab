import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Processed data
df = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')
df = df.dropna()
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map
car_map = shorten_categories(df.car.value_counts(), 10)
df['car'] = df['car'].map(car_map)
df.car.value_counts()
model_map = shorten_categories(df.model.value_counts(), 10)
df['model'] = df['model'].map(model_map)
df.model.value_counts()

st.set_page_config(page_title="DriveThru Deals – Data Explorer", page_icon="🚗", layout="wide")

st.title("Data Explorer 🔎📊")
st.write(
    "Welcome to the data exploration section. This page helps you understand how car prices "
    "relate to different features such as brand, model, body type, year, mileage and engine volume."
)
st.divider()
st.subheader("Filter the dataset")
st.write(
    """At the sidebar of this page you can filter the dataset as you wish to explore the specific data you want.
    All the values and plots will update dynamically based on your selected filters. The default state includes all the data."""
)

st.sidebar.header("Filters")
def multiselect_with_buttons(label: str, options, key: str):
    st.sidebar.write(label)
    if key not in st.session_state:
        st.session_state[key] = list(options)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select all", key=f"{key}_all"):
            st.session_state[key] = list(options)
    with col2:
        if st.button("Select none", key=f"{key}_none"):
            st.session_state[key] = []
    st.sidebar.multiselect(
        " ",
        options,
        key=key
    )
    return st.session_state[key]
brand_options = sorted(df["car"].unique())
selected_brands = multiselect_with_buttons("Brand", brand_options, "brand_filter")
model_options = sorted(df["model"].unique())
selected_models = multiselect_with_buttons("Model", model_options, "model_filter")
body_options = sorted(df["body"].unique())
selected_bodies = multiselect_with_buttons("Body type", body_options, "body_filter")
eng_options = sorted(df["engType"].unique())
selected_engTypes = multiselect_with_buttons("Engine type", eng_options, "eng_filter")
drive_options = sorted(df["drive"].unique())
selected_drive = multiselect_with_buttons("Drive type", drive_options, "drive_filter")

registration_values = sorted(df["registration"].unique())
registration_choice = st.sidebar.radio(
    "Registration",
    ["Both"] + registration_values,
    index=0
)
if registration_choice == "Both":
    selected_registration = registration_values
else:
    selected_registration = [registration_choice]
year_min, year_max = st.sidebar.slider("Year range",int(df.year.min()),int(df.year.max()),(int(df.year.min()), int(df.year.max())))
price_min, price_max = st.sidebar.slider("Price range",int(df.price.min()),int(df.price.max()),(int(df.price.min()), int(df.price.max())))
mileage_min, mileage_max = st.sidebar.slider("Mileage range",int(df.mileage.min()),int(df.mileage.max()),(int(df.mileage.min()), int(df.mileage.max())))
engV_min, engV_max = st.sidebar.slider("Engine volume (engV) range",float(df.engV.min()),float(df.engV.max()),(float(df.engV.min()), float(df.engV.max())))

outlier_option = st.sidebar.radio("Outliers",["Included", "Excluded"],index=0,help="If excluded, extreme values of mileage, engine volume, year and price will be removed.")

f = df[
    (df["car"].isin(selected_brands)) &
    (df["model"].isin(selected_models)) &
    (df["body"].isin(selected_bodies)) &
    (df["engType"].isin(selected_engTypes)) &
    (df["drive"].isin(selected_drive)) &
    (df["registration"].isin(selected_registration)) &
    (df["year"] >= year_min) & (df["year"] <= year_max) &
    (df["price"] >= price_min) & (df["price"] <= price_max) &
    (df["mileage"] >= mileage_min) & (df["mileage"] <= mileage_max) &
    (df["engV"] >= engV_min) & (df["engV"] <= engV_max)
]

if outlier_option == "Excluded":
    f = f[f["mileage"] <= 600]
    f = f[f["engV"] <= 7.5]
    f = f[f["year"] >= 1975]
    f = f[f["price"] >= 1000]
    f = f[f["price"] <= 100000]

if len(f) == 0:
    st.warning(
        "No data available for the selected filters.\n\n"
        "Try selecting at least one **Brand**, **Model**, **Body type**, "
        "**Engine type**, or changing the **year/price sliders**."
    )
    st.stop()

st.subheader("Overview of the filtered dataset")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of listings", f"{len(f):,}")
    st.metric("Year range", f"{int(f.year.min())} – {int(f.year.max())}")
    st.metric("Number of brands", f["car"].nunique())
with col2:
    st.metric("Median price", f"{int(f.price.median()):,}")
    st.metric("Mean price", f"{float(f.price.mean()):.2f}")
    st.metric("Number of body types", f["body"].nunique())
with col3:
    st.metric("Median mileage", f"{int(f.mileage.median()):,}")
    st.metric("Mean mileage", f"{float(f.mileage.mean()):.2f}")
    st.metric("Number of engine types", f["engType"].nunique())
with col4:
    st.metric("Median engine volume", f"{int(f.engV.median()):,}")
    st.metric("Mean engine volume", f"{float(f.engV.mean()):.2f}")
    st.metric("Registrated %", f"{(f["registration"].value_counts(normalize=True).get("yes", 0) * 100):.2f}%")

st.subheader("👀 Data preview")
st.write("Take a look at the dataset we are exploring.")
with st.expander("Show dataset"):
    st.dataframe(f.head(20))

#Distributions
st.divider()
st.subheader("📈 Distributions")
st.write("To understand the data, we must first know the distribution of the features.")
st.write("### Numeric distributions")
num1, num2 = st.columns(2)
with num1:
    st.write("#### Price distribution")
    st.write("(Lowering price filter recommended)")
    price_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("price", bin=alt.Bin(maxbins=50), title="Price"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(price_chart, use_container_width=True)
with num2:
    st.write("#### Mileage distribution")
    st.write("(Lowering mileage filter recommended)")
    mileage_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("mileage", bin=alt.Bin(maxbins=50), title="Mileage"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(mileage_chart, use_container_width=True)
num3, num4 = st.columns(2)
with num3:
    st.write("#### Engine Volume (engV) distribution")
    st.write("(Lowering engV filter recommended)")
    engV_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("engV", bin=alt.Bin(maxbins=50), title="Engine volume (L)"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(engV_chart, use_container_width=True)
with num4:
    st.write("#### Year distribution")
    st.write("(Raising year filter recommended)")
    year_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y="count()"
        )
        .properties(height=300)
    )
    st.altair_chart(year_chart, use_container_width=True)
st.write("### Categorical distributions")
cat1, cat2 = st.columns(2)
with cat1:
    st.write("#### Body type distribution")
    body_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("body", sort="-y", title="Body type"),
            y="count()",
            color=alt.Color("body", legend=None),
            tooltip=["body", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(body_chart, use_container_width=True)
with cat2:
    st.write("#### Engine type distribution")
    engType_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("engType", sort="-y", title="Engine type"),
            y="count()",
            color=alt.Color("engType", legend=None),
            tooltip=["engType", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(engType_chart, use_container_width=True)
cat3, cat4 = st.columns(2)
with cat3:
    st.write("#### Drive type distribution")
    drive_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("drive", sort="-y", title="Drive type"),
            y="count()",
            color=alt.Color("drive", legend=None),
            tooltip=["drive", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(drive_chart, use_container_width=True)
with cat4:
    st.write("#### Registration status distribution")
    reg_chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("registration", sort="-y", title="Registration"),
            y="count()",
            color=alt.Color("registration", legend=None),
            tooltip=["registration", "count()"]
        )
        .properties(height=300)
    )
    st.altair_chart(reg_chart, use_container_width=True)

# Average price by category
st.divider()
st.subheader("Average price by category")
st.write("Following we will look at the average price by some features.")
cat1, cat2 = st.columns(2)
with cat1:
    st.write("### Average price by body type")
    body_stats = f.groupby("body")["price"].mean().reset_index()
    body_chart = (
        alt.Chart(body_stats)
        .mark_bar()
        .encode(
            x=alt.X("body", sort="-y", title="Body type"),
            y=alt.Y("price", title="Average price"),
            color=alt.Color(
                "price",
                scale=alt.Scale(
                    domain=[body_stats["price"].min(), body_stats["price"].max()],
                    range=["green", "yellow", "red"]  # gradient
                ),
                legend=None
            ),
            tooltip=["body", "price"]
        )
        .properties(height=350)
    )
    st.altair_chart(body_chart, use_container_width=True)
with cat2:
    st.write("### Average price by brand")
    st.write("(Remove Bentley recommended)")
    brand_stats = f.groupby("car")["price"].mean().reset_index()
    brand_stats = brand_stats[brand_stats["car"] != "Bentley"]
    brand_chart = (
        alt.Chart(brand_stats)
        .mark_bar()
        .encode(
            x=alt.X("car", sort="-y", title="Brand"),
            y=alt.Y("price", title="Average price"),
            color=alt.Color(
                "price",
                scale=alt.Scale(
                    domain=[brand_stats["price"].min(), brand_stats["price"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["car", "price"]
        )
        .properties(height=350)
    )
    st.altair_chart(brand_chart, use_container_width=True)
cat3, cat4 = st.columns(2)
with cat3:
    st.write("### Average price by engine type")
    eng_stats = f.groupby("engType")["price"].mean().reset_index()

    eng_chart = (
        alt.Chart(eng_stats)
        .mark_bar()
        .encode(
            x=alt.X("engType", sort="-y", title="Engine type"),
            y=alt.Y("price", title="Average price"),
            color=alt.Color(
                "price",
                scale=alt.Scale(
                    domain=[eng_stats["price"].min(), eng_stats["price"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["engType", "price"]
        )
        .properties(height=350)
    )
    st.altair_chart(eng_chart, use_container_width=True)
with cat4:
    st.write("### Average price by drive type")
    drive_stats = f.groupby("drive")["price"].mean().reset_index()

    drive_chart = (
        alt.Chart(drive_stats)
        .mark_bar()
        .encode(
            x=alt.X("drive", sort="-y", title="Drive type"),
            y=alt.Y("price", title="Average price"),
            color=alt.Color(
                "price",
                scale=alt.Scale(
                    domain=[eng_stats["price"].min(), eng_stats["price"].max()],
                    range=["green", "yellow", "red"]
                ),
                legend=None
            ),
            tooltip=["drive", "price"]
        )
        .properties(height=350)
    )
    st.altair_chart(drive_chart, use_container_width=True)

st.write("### Average price by year")
year_stats = f.groupby("year")["price"].mean().reset_index()
year_chart = (
    alt.Chart(year_stats)
    .mark_bar()
    .encode(
        x=alt.X("year:O", sort="ascending", title="Year"),
        y=alt.Y("price", title="Average price"),
        color=alt.Color(
            "price",
            scale=alt.Scale(
                domain=[year_stats["price"].min(), year_stats["price"].max()],
                range=["green", "yellow", "red"]
            ),
            legend=None
        ),
        tooltip=["year", "price"]
    )
    .properties(height=350)
)
st.altair_chart(year_chart, use_container_width=True)

st.divider()
st.subheader("📈 Distributions of the features against the price")
st.write("""In these plots you can see each variable behaviour with respect to the price. 
        We stronlgy recommend to use the outliters filter in Excluded mode.""")
target = "price"
features = [x for x in f.columns if x not in ["car", "model", target]]
n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols  # round up
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
axes = axes.flatten()
for i, feature in enumerate(features):
    ax = axes[i]
    if f[feature].dtype == "object" or f[feature].nunique() < 10:
        sns.boxplot(x=feature, y=target, data=f, ax=ax)
    else:
        sns.scatterplot(x=feature, y=target, data=f, ax=ax)
    ax.set_title(f"{feature} vs {target}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j])
fig.tight_layout()
st.pyplot(fig)

st.divider()
st.subheader("Correlation Matrix")
st.write("""Here you can see the correlation matrix among all the fetures, it is important to focus on the price 
        row or column to see its correlation with the features. We also recommend to use the outliters filter in Excluded mode.""")

fcm = f.copy()
le_car = LabelEncoder()
fcm['car'] = le_car.fit_transform(fcm['car'])
le_body = LabelEncoder()
fcm['body'] = le_body.fit_transform(fcm['body'])
le_engType = LabelEncoder()
fcm['engType'] = le_engType.fit_transform(fcm['engType'])
le_drive = LabelEncoder()
fcm['drive'] = le_drive.fit_transform(fcm['drive'])
yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
fcm['registration'] = np.where(fcm['registration'].isin(yes_l), 1, 0)
fcm['registration'].value_counts()
fcm = fcm.drop(columns='model')

corr = fcm.corr().round(2)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="icefire", ax=ax)
ax.set_title("Correlation Heatmap", fontsize=10)
st.pyplot(fig)
