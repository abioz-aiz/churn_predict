import pickle

# from flask import Flask
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st



model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# streamlit app
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-image: url('https://images.unsplash.com/photo-1556740749-887f6717d7e4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080');
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    /* Main content area styling */
    .main {
        background: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
    }

    h1, h2, h3, h4 {
        color: #FFD700;
        text-shadow: 2px 2px 5px black;
    }

    /* Button styling */
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #FFA500;
        color: white;
    }

    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📋 Navigation")
    st.markdown(
        """
        - **Home**  
        - **About the App**  
        - **Contact**  
        """
    )
    st.write("Built with 💻 by [Your Name](https://github.com/your-profile)")

st.title("📈 Customer Churn Prediction")

st.markdown(
    '''
    Welcome to the **Customer Churn Prediction App**.
    Please fill in the details of the customer to predict churn probability
    '''
)

st.header('Customer Information')
with st.form(key='customer_form'):  
    customer = {
        "gender": st.selectbox("Gender", options=["female", "male"], index=0),
        "seniorcitizen": st.selectbox("Senior Citizen", options=[0, 1], index=0),
        "partner": st.selectbox("Partner", options=["yes", "no"], index=0),
        "dependents": st.selectbox("Dependents", options=["yes", "no"], index=1),
        "phoneservice": st.selectbox("Phone Service", options=["yes", "no"], index=1),
        "multiplelines": st.selectbox(
            "Multiple Lines",
            options=["yes", "no", "no_phone_service"],
            index=2,
        ),
        "internetservice": st.selectbox(
            "Internet Service", options=["dsl", "fiber_optic", "no"], index=0
        ),
        "onlinesecurity": st.selectbox("Online Security", options=["yes", "no", "no_internet_service"], index=1),
        "onlinebackup": st.selectbox("Online Backup", options=["yes", "no", "no_internet_service"], index=0),
        "deviceprotection": st.selectbox(
            "Device Protection", options=["yes", "no", "no_internet_service"], index=1
        ),
        "techsupport": st.selectbox("Tech Support", options=["yes", "no", "no_internet_service"], index=1),
        "streamingtv": st.selectbox("Streaming TV", options=["yes", "no", "no_internet_service"], index=1),
        "streamingmovies": st.selectbox("Streaming Movies", options=["yes", "no", "no_internet_service"], index=1),
        "contract": st.selectbox(
            "Contract", options=["month-to-month", "one_year", "two_year"], index=0
        ),
        "paperlessbilling": st.selectbox("Paperless Billing", options=["yes", "no"], index=0),
        "paymentmethod": st.selectbox(
            "Payment Method",
            options=["electronic_check", "mailed_check", "bank_transfer (automatic)", "credit_card (automatic)"],
            index=0,
        ),
        "tenure": st.number_input("Tenure (in months)", min_value=0, value=1, step=1),
        "monthlycharges": st.number_input("Monthly Charges", min_value=0.0, value=29.85, step=0.01),
        "totalcharges": st.number_input("Total Charges", min_value=0.0, value=29.85, step=0.01),
    }
    submit_button = st.form_submit_button(label='Predict Churn')


if submit_button:
    # Transform input data and make prediction
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    # Display the results
    st.subheader("Prediction Results")
    st.markdown("<div class = 'metric-container'>", unsafe_allow_html=True)
    st.write(f"Churn Probability: {y_pred:.2f}")
    st.write(f"Churn Prediction: {'Yes' if churn else 'No'}")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    ---
    """
)