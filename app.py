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

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title and Description ---
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
This interactive web application uses a Random Forest model to detect fraudulent credit card transactions.
The model was trained on a dataset from Kaggle, which has been preprocessed and balanced using SMOTE to handle class imbalance.
Enter transaction details in the sidebar to get a real-time prediction.
""")

# --- Helper Function to Load and Prepare Data ---
@st.cache_data
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and prepares the dataset."""
    try:
        df = pd.read_csv(filepath)
        # For demonstration, we'll use a smaller sample to speed up processing.
        # In a real scenario, you would use the full dataset.
        df = df.sample(n=50000, random_state=42)

        # Drop 'Time' as it's often not a useful feature without significant engineering
        df = df.drop(['Time'], axis=1)

        # Scale the 'Amount' feature
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df = df.drop(['Amount'], axis=1)

        X = df.drop('Class', axis=1)
        y = df['Class']

        return X, y, df
    except FileNotFoundError:
        st.error(f"Dataset file not found at {filepath}. Please make sure 'creditcard.csv' is in the same directory.")
        return None, None, None

# --- Model Training Function ---
@st.cache_resource
def train_model(X, y):
    """Handles class imbalance with SMOTE and trains a RandomForestClassifier."""
    st.write("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    st.write("Splitting data and training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Return the total number of rows used for training (after SMOTE)
    return model, accuracy, report, len(X_res)

# --- Define file paths ---
data_file = 'creditcard.csv'
model_file = 'fraud_detection_model.pkl'

# --- Feature columns (needed for creating the input form) ---
feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                'scaled_amount']

# --- Main Application Logic ---

# Try to load the pre-trained model and its metrics
model = None
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        loaded_object = pickle.load(f)

    # Add a check to handle both old (model-only) and new (dictionary) .pkl files
    if isinstance(loaded_object, dict):
        # New format with metrics
        model = loaded_object['model']
        accuracy = loaded_object['accuracy']
        report = loaded_object['report']
        training_rows = loaded_object['training_rows']
        st.success("Loaded pre-trained model and metrics successfully!")

        # --- Display Model Performance Metrics ---
        st.markdown("### Model Performance Metrics")
        st.write(f"The model was trained on a balanced dataset of **{training_rows:,}** transactions (after SMOTE).")

        col1, col2, col3 = st.columns(3)
        # Check if '1' (fraud class) exists in the report, for safety
        if '1' in report:
            recall_fraud = report['1']['recall']
            precision_fraud = report['1']['precision']
        else:
            recall_fraud = 0.0
            precision_fraud = 0.0

        col1.metric("Overall Accuracy", f"{accuracy:.2%}")
        col2.metric("Fraud Recall", f"{recall_fraud:.2%}", help="Of all actual fraud cases, what percentage did the model correctly identify?")
        col3.metric("Fraud Precision", f"{precision_fraud:.2%}", help="Of all transactions the model flagged as fraud, what percentage were actually fraudulent?")

        with st.expander("View Full Classification Report"):
            st.json(report)
    else:
        # Old format, just the model was saved
        model = loaded_object
        st.warning("Loaded a model file without metrics. To see performance data, please delete the `.pkl` file, run the app locally to regenerate it, and then redeploy.")

else:
    st.warning("Pre-trained model not found. The app will now attempt to load data and train a new model. This is for local development only.")
    # This block is for local development if the model doesn't exist
    if not os.path.exists(data_file):
        st.error("`creditcard.csv` not found. Please download it from Kaggle for local model training.")
        st.markdown("[Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    else:
        # Load Data
        with st.spinner('Loading and preparing data...'):
            X, y, df = load_and_prepare_data(data_file)

        if X is not None:
            st.header("Dataset Overview (Local Development)")
            st.write("Here's a preview of the dataset (after sampling and scaling):")
            st.dataframe(df.head())

            # Display charts and training info locally
            st.subheader("Class Distribution")
            class_dist = df['Class'].value_counts()
            st.bar_chart(class_dist)

            with st.spinner('Training model... This might take a few minutes.'):
                model, accuracy, report, training_rows = train_model(X, y)
                
                # Save the model AND its metrics in a dictionary
                model_data_to_save = {
                    'model': model,
                    'accuracy': accuracy,
                    'report': report,
                    'training_rows': training_rows
                }
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data_to_save, f)

            st.success("Model trained and saved successfully!")
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write("**Classification Report:**")
            st.json(report)

# --- Sidebar for User Input (Only if model is loaded) ---
if model is not None:
    st.sidebar.header("Check a Transaction")
    st.sidebar.markdown("Enter the transaction features to predict if it's fraudulent.")

    input_features = {}
    for feature in feature_cols:
        input_features[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=0.0, format="%.6f")

    predict_button = st.sidebar.button("Predict Fraud", use_container_width=True)

    if predict_button:
        # Create a DataFrame from the user's input
        input_df = pd.DataFrame([input_features])

        # Ensure the order of columns matches the training data
        input_df = input_df[feature_cols]

        # Make a prediction
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

# --- Footer ---
st.markdown("---")
st.markdown("Built by Pradyumn Kumar Shukla @ 2025")
st.markdown("Data Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
