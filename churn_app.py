import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Healthy Meals Churn Predictor", page_icon="📉")

model = joblib.load("churn_model.pkl")

numeric_features = ['total_num_sessions', 'gross_session_length', 'active_days',
                     'active_quarters', 'avg_sessions_per_active_quarter',
                     'avg_session_length', 'sessions_per_active_day',
                     'age', 'tech_comfort_score']
categorical_features = ['education', 'income_level', 'device_type']

# Pull the exact categories the encoder was fitted on -- guarantees the dropdown
# options match training exactly, no hardcoded strings to get wrong.
cat_encoder = model.named_steps['preprocess'].named_transformers_['cat']
cat_options = dict(zip(categorical_features, cat_encoder.categories_))

st.title("Healthy Meals — Churn Probability Predictor")
st.write("Enter a customer's activity and demographic profile to get a predicted probability of churn.")

col1, col2 = st.columns(2)

with col1:
    total_num_sessions = st.number_input("Total sessions (prior year)", min_value=0, value=30)
    gross_session_length = st.number_input("Gross session length, minutes (prior year)", min_value=0, value=800)
    active_days = st.number_input("Active days (prior year)", min_value=0, max_value=365, value=2)
    active_quarters = st.number_input("Active quarters (0-4)", min_value=0, max_value=4, value=2)
    avg_sessions_per_active_quarter = st.number_input("Avg sessions per active quarter", min_value=0.0, value=15.0)

with col2:
    avg_session_length = st.number_input("Avg session length, minutes", min_value=0.0, value=25.0)
    sessions_per_active_day = st.number_input("Sessions per active day", min_value=0.0, value=15.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=36)
    tech_comfort_score = st.slider("Tech comfort score", min_value=1, max_value=5, value=3)

education = st.selectbox("Education", options=cat_options['education'])
income_level = st.selectbox("Income level", options=cat_options['income_level'])
device_type = st.selectbox("Device type", options=cat_options['device_type'])

if st.button("Predict churn probability"):
    input_df = pd.DataFrame([{
        'total_num_sessions': total_num_sessions,
        'gross_session_length': gross_session_length,
        'active_days': active_days,
        'active_quarters': active_quarters,
        'avg_sessions_per_active_quarter': avg_sessions_per_active_quarter,
        'avg_session_length': avg_session_length,
        'sessions_per_active_day': sessions_per_active_day,
        'age': age,
        'tech_comfort_score': tech_comfort_score,
        'education': education,
        'income_level': income_level,
        'device_type': device_type,
    }])

    prob = model.predict_proba(input_df)[0, 1]
    st.metric("Predicted churn probability", f"{prob:.1%}")

    if prob < 0.10:
        st.success("Low risk")
    elif prob < 0.25:
        st.warning("Moderate risk")
    else:
        st.error("High risk — consider proactive outreach")
