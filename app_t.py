import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Random Forest Classifier", layout="centered")


def load_css(file):
    if os.path.exists(file):
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css('style.css')

st.markdown(
    """
    <div class="card">
    <h1>Random Forest Classifier</h1>
    <p>Train and evaluate a Random Forest model on a tabular CSV dataset. Upload a CSV or use a local path.</p>
    </div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def try_load_path(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_uploaded(file):
    try:
        return pd.read_csv(file)
    except Exception:
        return None


# Sidebar: data selection and model params
st.sidebar.header("Data & Model Settings")
upload = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default_path = os.path.exists(r"C:\\Users\\mrdee\\Downloads\\RF datasets.csv")

if upload is None and use_default_path:
    path = r"C:\Users\mrdee\Downloads\RF datasets.csv"
else:
    path = None

if upload is not None:
    df = load_uploaded(upload)
elif path:
    df = try_load_path(path)
else:
    df = None

if df is None:
    st.warning("No dataset loaded. Upload a CSV or place the dataset at C:/Users/mrdee/Downloads/RF datasets.csv")
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2>Dataset Preview</h2>', unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# let user choose target
all_columns = df.columns.tolist()
default_target = 'target_left_company' if 'target_left_company' in all_columns else all_columns[-1]
target = st.sidebar.selectbox("Target column", all_columns, index=all_columns.index(default_target))

# choose categorical columns (auto-detect)
auto_cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = st.sidebar.multiselect("Categorical columns to One-Hot Encode", options=auto_cats, default=auto_cats)

# model hyperparams
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, step=0.05)
random_state = int(st.sidebar.number_input("Random state", value=42))

# Prepare X, y
X = df.drop(columns=[target])
y = df[target]

# Keep track of feature order
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# enforce categorical cols exist in X
cat_cols = [c for c in cat_cols if c in X.columns]

# Fit OneHotEncoder if needed
if len(cat_cols) > 0:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    ohe.fit(X[cat_cols])
    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
    X_ohe = pd.DataFrame(ohe.transform(X[cat_cols]), columns=ohe_cols, index=X.index)
    X_proc = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), X_ohe.reset_index(drop=True)], axis=1)
else:
    ohe = None
    X_proc = X.copy()

# train/test split
x_train, x_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=random_state)

# Train model
@st.cache_data
def train_model(x_train, y_train, n_estimators, random_state):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(x_train, y_train)
    return clf

model = train_model(x_train, y_train, n_estimators, random_state)

# Predict & metrics
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)

st.markdown('<div class="card"><h3>Model Performance</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.3f}")
col2.write("")

st.write("**Classification Report**")
st.text(report)

st.write("**Confusion Matrix**")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Feature importance
importances = model.feature_importances_
feat_names = X_proc.columns.tolist()
fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)

st.markdown('<div class="card"><h3>Feature Importance</h3>', unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(8, min(6, 0.2 * len(fi))))
fi.head(20).plot(kind='barh', ax=ax2)
ax2.invert_yaxis()
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction UI
st.markdown('<div class="card"><h3>Make a Prediction</h3>', unsafe_allow_html=True)
with st.form(key='predict_form'):
    inputs = {}
    for col in numeric_cols:
        minv = float(df[col].min()) if pd.notna(df[col].min()) else 0.0
        maxv = float(df[col].max()) if pd.notna(df[col].max()) else minv + 1.0
        meanv = float(df[col].mean()) if pd.notna(df[col].mean()) else minv
        inputs[col] = st.number_input(col, value=meanv, min_value=minv, max_value=maxv)

    for col in cat_cols:
        options = sorted(df[col].dropna().unique().tolist())
        default = options[0] if len(options) > 0 else ''
        inputs[col] = st.selectbox(col, options, index=0)

    submit = st.form_submit_button('Predict')

if submit:
    # build single-row DataFrame
    row = pd.DataFrame([inputs])
    # process categorical
    if ohe is not None and len(cat_cols) > 0:
        row_ohe = pd.DataFrame(ohe.transform(row[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        row_proc = pd.concat([row.drop(columns=cat_cols).reset_index(drop=True), row_ohe.reset_index(drop=True)], axis=1)
    else:
        row_proc = row.copy()

    # ensure same column order
    for c in X_proc.columns:
        if c not in row_proc.columns:
            row_proc[c] = 0
    row_proc = row_proc[X_proc.columns]

    pred = model.predict(row_proc)[0]
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(row_proc)[0]
        proba_str = ', '.join([f'{p:.3f}' for p in proba])
    else:
        proba_str = 'N/A'

    st.markdown(f"**Predicted:** {pred}")
    st.markdown(f"**Probabilities:** {proba_str}")
    
st.markdown('</div>', unsafe_allow_html=True)

st.caption('Dependencies: streamlit, pandas, scikit-learn, seaborn, matplotlib, numpy')
