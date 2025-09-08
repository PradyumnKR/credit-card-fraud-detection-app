import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

# Set up the Streamlit page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and a short intro
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
This app helps you spot fraudulent credit card transactions using a machine learning model.
The model was trained on real data from Kaggle, and we've balanced the classes to improve accuracy.
Just enter the transaction details on the left to see if it's flagged as fraud.
""")

# Function to load and clean up the data
@st.cache_data
def load_and_prepare_data(filepath):
    """Read the CSV, sample it for speed, and do some basic feature engineering."""
    try:
        df = pd.read_csv(filepath)
        # We'll use a smaller chunk of data to keep things fast for demo purposes.
        df = df.sample(n=50000, random_state=42)

        # Remove the 'Time' column since it's not very useful here
        df = df.drop(['Time'], axis=1)

        # Standardize the 'Amount' column
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df = df.drop(['Amount'], axis=1)

        X = df.drop('Class', axis=1)
        y = df['Class']

        return X, y, df
    except FileNotFoundError:
        st.error(f"Dataset file not found at {filepath}. Please make sure 'creditcard.csv' is in the same directory.")
        return None, None, None

# Function to train the model
@st.cache_resource
def train_model(X, y):
    """Balance the classes and train a Random Forest model."""
    st.write("Balancing the data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    st.write("Splitting the data and fitting the model...")
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Check how well the model does
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report

# Main part of the app starts here
data_file = 'creditcard.csv'
model_file = 'fraud_detection_model.pkl'

# Make sure the dataset is available
if not os.path.exists(data_file):
    st.warning("Couldn't find 'creditcard.csv'. Please download it from Kaggle and put it in this folder.")
    st.markdown("[Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
else:
    # Load the data
    with st.spinner('Loading and preparing data...'):
        X, y, df = load_and_prepare_data(data_file)

    if X is not None:
        st.header("Dataset Overview")
        st.write("Here's a quick look at the data after sampling and scaling:")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Distribution")
            class_dist = df['Class'].value_counts()
            st.bar_chart(class_dist)
            st.write(f"**Legitimate Transactions (0):** {class_dist.get(0, 0)}")
            st.write(f"**Fraudulent Transactions (1):** {class_dist.get(1, 0)}")

        # Either load the model or train a new one
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            st.success("Loaded pre-trained model.")
        else:
            with st.spinner('Training model... This could take a few minutes.'):
                model, accuracy, report = train_model(X, y)
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            st.success("Model trained successfully!")
            with col2:
                st.subheader("Model Performance")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write("**Classification Report:**")
                st.json(report)

        # Sidebar for entering transaction details

        st.sidebar.header("Check a Transaction")
        st.sidebar.markdown("Fill in the details below to see if the transaction looks suspicious.")

        input_features = {}
        # There are 29 features (V1-V28 and scaled_amount)
        for feature in X.columns:
            input_features[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=0.0, format="%.6f")

        predict_button = st.sidebar.button("Predict Fraud", use_container_width=True)

        if predict_button:
            # Turn the user's input into a DataFrame
            input_df = pd.DataFrame([input_features])

            # Make sure the columns are in the right order
            input_df = input_df[X.columns]

            # Get the prediction from the model
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.sidebar.subheader("Prediction Result")
            if prediction[0] == 1:
                st.sidebar.error("Prediction: Fraudulent Transaction")
            else:
                st.sidebar.success("Prediction: Legitimate Transaction")

            st.sidebar.write("**Prediction Probability:**")
            st.sidebar.write(f"Legitimate: {prediction_proba[0][0]:.4f}")
            st.sidebar.write(f"Fraudulent: {prediction_proba[0][1]:.4f}")

# Footer
st.markdown("---")
st.markdown("Built by Pradyumn Kumar Shukla @ 2025")
