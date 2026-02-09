import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="California Housing - Random Forest", layout="centered")


def load_css(file):
    if os.path.exists(file):
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css('style.css')

st.markdown(
    """
    <div class="card">
    <h1>California Housing — Random Forest Regressor</h1>
    <p>Train a Random Forest to predict median house value using the California housing dataset from scikit-learn.</p>
    </div>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame
    return df


df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2>Dataset Preview</h2>', unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Model & Data")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, step=0.05)
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("max_depth (None=0)", 0, 50, 0, step=1)
if max_depth == 0:
    max_depth = None
random_state = int(st.sidebar.number_input("Random state", value=42))

# Prepare data
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

@st.cache_data
def train_model(x_train, y_train, n_estimators, max_depth, random_state):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)
    return model

model = train_model(x_train, y_train, n_estimators, max_depth, random_state)

# Evaluate
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.markdown('<div class="card"><h3>Model Performance</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3, col4 = st.columns(2)
col3.metric("R2", f"{r2:.3f}")
col4.write("")

st.markdown("**Actual vs Predicted (sample)**")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Feature importance
importances = model.feature_importances_
feat_names = X.columns.tolist()
fi = pd.Series(importances, index=feat_names).sort_values(ascending=True)

st.markdown('<div class="card"><h3>Feature Importance</h3>', unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.3 * len(fi))))
fi.plot(kind='barh', ax=ax2)
ax2.set_xlabel('Importance')
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction UI
st.markdown('<div class="card"><h3>Make a Prediction</h3>', unsafe_allow_html=True)
with st.form('predict_form'):
    inputs = {}
    for col in X.columns:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
        # sensible step for sliders
        step = (col_max - col_min) / 100 if (col_max - col_min) > 0 else 0.1
        inputs[col] = st.number_input(col, value=col_mean, min_value=col_min, max_value=col_max, step=step)
    submit = st.form_submit_button('Predict')

if submit:
    row = pd.DataFrame([inputs])
    pred = model.predict(row)[0]
    st.markdown(f"**Predicted Median House Value:** {pred:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

st.caption('Dependencies: streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn')
