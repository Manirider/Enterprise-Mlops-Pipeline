import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import os

# Set page config for a premium feel
st.set_page_config(
    page_title="Enterprise MLOps Pipeline Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Main Theme */
    :root {
        --primary-color: #3b82f6;
        --background-color: #f8fafc;
        --card-bg: #ffffff;
        --text-main: #1e293b;
        --text-muted: #64748b;
    }
    
    .stApp {
        background-color: var(--background-color);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-main);
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Metrics and Cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        color: var(--text-muted);
        font-size: 0.875rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: var(--primary-color);
        font-size: 1.875rem;
        font-weight: 700;
    }
    
    /* Predict Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Prediction Result */
    .prediction-result {
        text-align: center;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    .prediction-positive {
        background-color: #dcfce7;
        border: 1px solid #bbf7d0;
        color: #166534;
    }
    .prediction-negative {
        background-color: #fee2e2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_resource
def load_pipeline_artifacts():
    base_dir = Path(__file__).resolve().parent
    
    # 1. Load Model
    model_path = base_dir / 'models' / 'model.joblib'
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please run the DVC pipeline.")
        return None, None, None, None
        
    model = joblib.load(model_path)
    
    # 2. Load Processed Data to fit OrdinalEncoder
    # (Since the pipeline doesn't serialize the encoder, we fit it on the fly)
    data_path = base_dir / 'data' / 'processed.csv'
    if not data_path.exists():
        st.error(f"Processed data not found at {data_path}. Please run the DVC pipeline.")
        return None, None, None, None
        
    df = pd.read_csv(data_path)
    target_col = 'income'
    if target_col in df.columns:
        X_df = df.drop(columns=[target_col])
    else:
        X_df = df
        
    cat_cols = X_df.select_dtypes(include='object').columns.tolist()
    num_cols = X_df.select_dtypes(exclude='object').columns.tolist()
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        encoder.fit(X_df[cat_cols])
        
    # Get unique values for dropdowns
    unique_vals = {col: sorted(X_df[col].dropna().unique().tolist()) for col in cat_cols}
    
    # Load Metrics
    metrics_path = base_dir / 'metrics' / 'scores.json'
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return model, encoder, unique_vals, metrics, num_cols, cat_cols

model, encoder, unique_vals, metrics, num_cols, cat_cols = load_pipeline_artifacts()

if model is None:
    st.stop()

# --- Helper functions for feature engineering ---
def engineer_features(input_dict):
    df = pd.DataFrame([input_dict])
    
    # Engineer interactions exactly as in src/featurize.py
    df['capital_net'] = df['capital_gain'] - df['capital_loss']
    df['has_capital'] = ((df['capital_gain'] > 0) | (df['capital_loss'] > 0)).astype(int)
    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    df['hours_bin'] = pd.cut(df['hours_per_week'], bins=[0, 35, 45, 100], labels=[0, 1, 2]).astype(int)
    
    # Final list of expected numeric columns (base + engineered)
    expected_num_cols = num_cols + ['capital_net', 'has_capital', 'age_bin', 'hours_bin']
    
    # Encode categorical features
    if cat_cols and encoder is not None:
        X_cat = encoder.transform(df[cat_cols])
        X_num = df[expected_num_cols].to_numpy(dtype=np.float64)
        X_final = np.hstack([X_num, X_cat])
    else:
        X_final = df[expected_num_cols].to_numpy(dtype=np.float64)
        
    feature_names = expected_num_cols + cat_cols
    
    # Return as DataFrame to prevent LightGBM warnings about column names
    return pd.DataFrame(X_final, columns=feature_names)


# --- Layout ---

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8636/8636906.png", width=60)
    st.markdown("### MLOps Pipeline Status")
    st.success("🟢 Pipeline Active")
    
    st.markdown("---")
    st.markdown("#### Model Performance")
    if metrics:
        st.markdown(f"**Accuracy:** `{metrics.get('test_accuracy', 0):.4f}`")
        st.markdown(f"**ROC AUC:** `{metrics.get('test_roc_auc', 0):.4f}`")
        st.markdown(f"**Precision:** `{metrics.get('test_precision', 0):.4f}`")
        st.markdown(f"**Recall:** `{metrics.get('test_recall', 0):.4f}`")
        st.markdown(f"**F1 Score:** `{metrics.get('test_f1', 0):.4f}`")
    else:
        st.warning("Metrics not available.")
        
    st.markdown("---")
    st.markdown("#### Pipeline Info")
    st.markdown("Engine: **LightGBM**")
    st.markdown("Tracking: **DVC**")
    st.markdown("Features: **18** (incl. interactions)")

# Main Area
st.title("Enterprise AI Income Predictor")
st.markdown("Real-time inference engine powered by our modular MLOps architecture. Adjust the demographic and financial parameters below to predict income bracket.")

with st.form("prediction_form"):
    st.markdown("### Profile Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=100, value=30, step=1)
        education = st.selectbox("Education", unique_vals.get('education', []))
        education_num = st.number_input("Education Years", min_value=1, max_value=16, value=10, step=1)
        marital_status = st.selectbox("Marital Status", unique_vals.get('marital_status', []))
        
    with col2:
        workclass = st.selectbox("Workclass", unique_vals.get('workclass', []))
        occupation = st.selectbox("Occupation", unique_vals.get('occupation', []))
        relationship = st.selectbox("Relationship", unique_vals.get('relationship', []))
        hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40, step=1)
        
    with col3:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, value=0, step=100)
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, value=0, step=100)
        race = st.selectbox("Race", unique_vals.get('race', []))
        sex = st.selectbox("Sex", unique_vals.get('sex', []))
        
    native_country = st.selectbox("Native Country", unique_vals.get('native_country', []))
    
    # We also need fnlwgt, though it's typically just a demographic weight. Use median/mean.
    # In real world, we might not ask user for fnlwgt. We'll pass a default.
    fnlwgt = 189778.0 # typical median value from adult dataset
    
    submit = st.form_submit_button("Generate Prediction")

if submit:
    # Prepare input dict
    input_data = {
        'age': age,
        'fnlwgt': fnlwgt,
        'education_num': education_num,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'workclass': workclass,
        'education': education,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'native_country': native_country
    }
    
    try:
        # Engineer features
        X_inference = engineer_features(input_data)
        
        # Predict
        prediction_prob = model.predict_proba(X_inference)[0][1]
        prediction_class = model.predict(X_inference)[0]
        
        # Render Result
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        if prediction_class == 1:
            st.markdown(f'''
            <div class="prediction-result prediction-positive">
                <h2 style="color: #166534; margin: 0;">> $50,000 / year</h2>
                <p style="margin-top: 10px; font-size: 1.1rem;">Confidence: <strong>{prediction_prob:.1%}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'''
            <div class="prediction-result prediction-negative">
                <h2 style="color: #991b1b; margin: 0;"><= $50,000 / year</h2>
                <p style="margin-top: 10px; font-size: 1.1rem;">Probability of high income: <strong>{prediction_prob:.1%}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
