
# ============================================================
# SMART HEALTHCARE DIAGNOSIS ASSISTANT
# STREAMLIT APPLICATION
# TF-IDF + SUPPORT VECTOR MACHINE
# ============================================================

import os
import re
import string
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT DIRECTORY AND FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "healthcare_svm_pipeline.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.pkl")
LABELS_FILE = os.path.join(BASE_DIR, "class_labels.pkl")
DATASET_INFO_FILE = os.path.join(BASE_DIR, "dataset_info.pkl")
DATASET_FILE = os.path.join(BASE_DIR, "DataSet_balanced.csv")

TEXT_COLUMN = "Description"
TARGET_COLUMN = "Disease_Name"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Healthcare Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# 3. NLTK RESOURCE SETUP
# ============================================================

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return True


setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)




# ============================================================
# 5. DISEASE-TO-DEPARTMENT SUGGESTION
# ============================================================

DISEASE_DEPARTMENT = {
    "Malaria": "General Medicine / Internal Medicine",
    "Dengue": "General Medicine / Internal Medicine",
    "Typhoid Fever": "General Medicine / Internal Medicine",
    "Tuberculosis": "Pulmonology",
    "COVID-19": "Pulmonology",
    "Chikungunya": "General Medicine / Internal Medicine",
    "Cholera": "Gastroenterology",
    "Hepatitis A": "Gastroenterology / Hepatology",
    "Chickenpox": "Dermatology",
    "Measles": "General Medicine / Internal Medicine",
    "Mumps": "General Medicine / ENT",
    "Whooping Cough": "Pulmonology",
    "Common Cold": "General Medicine / ENT",
    "Influenza": "General Medicine",
    "Migraine": "Neurology",
    "Allergic Rhinitis": "ENT / Allergy and Immunology",
    "Gastroenteritis": "Gastroenterology",
    "Asthma": "Pulmonology",
    "Type 2 Diabetes": "Endocrinology / Diabetology",
    "Hypertension": "Cardiology / General Medicine",
    "Urinary Tract Infection": "Urology",
    "Iron Deficiency Anemia": "Hematology",
    "Acid Reflux": "Gastroenterology",
    "Dermatitis": "Dermatology",
    "Conjunctivitis": "Ophthalmology",
    "Tension Headache": "Neurology / General Medicine",
    "Sinusitis": "ENT",
    "Pneumonia": "Pulmonology",
    "Appendicitis": "General Surgery",
    "Kidney Stone": "Urology"
}


def get_suggested_department(disease):
    """Return the educational department suggestion for a predicted disease."""
    return DISEASE_DEPARTMENT.get(
        str(disease).strip(),
        "General Medicine (Initial Consultation)"
    )


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


# ============================================================
# 7. LOAD MODEL METRICS
# ============================================================

@st.cache_resource
def load_metrics():

    with open(METRICS_FILE, "rb") as file:
        return pickle.load(file)


# ============================================================
# 8. LOAD CLASS LABELS
# ============================================================

@st.cache_resource
def load_labels():

    with open(LABELS_FILE, "rb") as file:
        return pickle.load(file)


# ============================================================
# 9. LOAD DATASET INFORMATION
# ============================================================

@st.cache_resource
def load_dataset_info():

    if os.path.exists(DATASET_INFO_FILE):

        with open(DATASET_INFO_FILE, "rb") as file:
            return pickle.load(file)

    return {}


# ============================================================
# 10. APPLICATION HEADER
# ============================================================

st.title("🏥 Smart Healthcare Diagnosis Assistant")

st.subheader(
    "AI-Based Disease Prediction Using Support Vector Machine"
)

st.write(
    """
    Enter the patient's symptoms or clinical description.
    The trained TF-IDF and Support Vector Machine model
    analyzes the text and predicts a disease category.
    """
)

st.info(
    "This application is an educational machine-learning "
    "demonstration, not a medical diagnostic service."
)


# ============================================================
# 11. CHECK REQUIRED MODEL FILES
# ============================================================

required_files = [
    MODEL_FILE,
    METRICS_FILE,
    LABELS_FILE
]

missing_files = [
    os.path.basename(file)
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("Trained model files were not found.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.warning(
        "First run model_train.py to train the SVM model."
    )

    st.code("python model_train.py")

    st.stop()


# ============================================================
# 12. LOAD MODEL FILES
# ============================================================

try:

    model = load_model()
    metrics = load_metrics()
    labels = load_labels()
    dataset_info = load_dataset_info()

except Exception as e:

    st.error("Error while loading the trained model files.")
    st.exception(e)
    st.stop()


# ============================================================
# 13. SIDEBAR PROJECT DETAILS
# ============================================================

st.sidebar.title("🏥 Project Details")

st.sidebar.success("Support Vector Machine")

st.sidebar.write("### Machine Learning Pipeline")

st.sidebar.code(
    """
Patient Description
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Linear SVM
       ↓
Predicted Disease
       ↓
Suggested Department
    """
)

st.sidebar.write("**Dataset:**")
st.sidebar.info("DataSet_balanced.csv")

st.sidebar.write("**Input Feature:**")
st.sidebar.info(TEXT_COLUMN)

st.sidebar.write("**Target Variable:**")
st.sidebar.info(TARGET_COLUMN)

st.sidebar.write("**Algorithm:**")
st.sidebar.info("LinearSVC")


# ============================================================
# 14. SIDEBAR MODEL PERFORMANCE
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("📊 SVM Performance")

accuracy = metrics.get("accuracy")
precision = metrics.get("precision")
recall = metrics.get("recall")
f1 = metrics.get("f1_score")


def show_metric(container, name, value):

    if value is not None:
        container.metric(name, f"{value * 100:.2f}%")
    else:
        container.metric(name, "N/A")


show_metric(st.sidebar, "Accuracy", accuracy)
show_metric(st.sidebar, "Precision", precision)
show_metric(st.sidebar, "Recall", recall)
show_metric(st.sidebar, "F1 Score", f1)


# ============================================================
# 15. MAIN MODEL PERFORMANCE
# ============================================================

st.divider()
st.header("📊 Model Performance")

if accuracy is not None:

    col1, col2, col3, col4 = st.columns(4)

    show_metric(col1, "Accuracy", accuracy)
    show_metric(col2, "Precision", precision)
    show_metric(col3, "Recall", recall)
    show_metric(col4, "F1 Score", f1)

else:

    st.warning("Test-set performance metrics are unavailable.")


# ============================================================
# 16. DATASET INFORMATION
# ============================================================

st.header("📁 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:

    total_records = dataset_info.get(
        "total_training_records", "N/A"
    )

    st.metric("Total Records", total_records)

with col2:

    number_diseases = dataset_info.get(
        "number_of_diseases", len(labels)
    )

    st.metric("Disease Classes", number_diseases)

with col3:

    dataset_shape = dataset_info.get("dataset_shape")

    if dataset_shape is not None:
        st.metric("Dataset Columns", dataset_shape[1])
    else:
        st.metric("Dataset Columns", "N/A")

st.write("**Dataset filename:** DataSet_balanced.csv")
st.write(f"**Input column:** {TEXT_COLUMN}")
st.write(f"**Target column:** {TARGET_COLUMN}")


# ============================================================
# 17. PATIENT SYMPTOM INPUT AND DISEASE PREDICTION
# ============================================================

st.divider()

st.header("🩺 Enter Patient Symptoms")

st.write(
    """
    Enter the patient's symptoms or clinical description
    in the box below and click Predict Disease.
    """
)

# Form keeps input and prediction button together
with st.form("disease_prediction_form"):

    user_input = st.text_area(
        "Patient Symptoms / Clinical Description",

        placeholder=(
            "Example: fever, headache, chills, "
            "body pain, weakness and sweating"
        ),

        height=180
    )

    predict_button = st.form_submit_button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 18. DISPLAY PREDICTION RESULT
# ============================================================

if predict_button:

    if not user_input.strip():

        st.warning("Please enter the patient's symptoms.")

    else:

        # Preprocess patient input
        processed_input = preprocess_text(user_input)

        if not processed_input.strip():

            st.error(
                "Please enter valid symptoms containing usable words."
            )

        else:

            try:

                # Predict disease using trained SVM model
                prediction = model.predict([processed_input])

                predicted_disease = str(prediction[0])

                st.divider()

                st.header("🔍 Prediction Result")

                st.success(
                    f"Predicted Disease: {predicted_disease}"
                )

                # Suggest a relevant medical department
                suggested_department = get_suggested_department(
                    predicted_disease
                )

                st.info(
                    f"Suggested Department: {suggested_department}"
                )

                # Show entered symptoms
                st.subheader("📝 Patient Information")

                st.write("**Entered Symptoms:**")
                st.write(user_input)

                # Show processed text
                with st.expander("View Text Preprocessing"):

                    st.write("**Original Input:**")
                    st.write(user_input)

                    st.write("**Processed Input:**")
                    st.write(processed_input)

                # Show SVM decision scores when available
                try:

                    if hasattr(model, "named_steps"):

                        svm_model = model.named_steps.get("svm")
                        tfidf_model = model.named_steps.get("tfidf")

                        if svm_model is not None and tfidf_model is not None:

                            transformed_input = tfidf_model.transform(
                                [processed_input]
                            )

                            decision_scores = svm_model.decision_function(
                                transformed_input
                            )

                            scores = np.asarray(decision_scores)
                            classes = svm_model.classes_

                            if (
                                scores.ndim == 2
                                and scores.shape[1] == len(classes)
                            ):

                                score_df = pd.DataFrame({
                                    "Disease": classes,
                                    "SVM Decision Score": scores[0]
                                })

                                score_df = score_df.sort_values(
                                    by="SVM Decision Score",
                                    ascending=False
                                ).reset_index(drop=True)

                                st.subheader("📈 Top SVM Predictions")

                                st.caption(
                                    "SVM decision scores are not probabilities "
                                    "or confirmed diagnostic confidence."
                                )

                                st.dataframe(
                                    score_df.head(10),
                                    use_container_width=True,
                                    hide_index=True
                                )

                except Exception:

                    st.info(
                        "Additional SVM decision scores are unavailable."
                    )

                # Medical disclaimer
                st.warning(
                    """
                    **Educational Project Disclaimer**

                    This application is developed for educational
                    and demonstration purposes.

                    The predicted disease is a machine-learning
                    output based on the training dataset. It is
                    not a confirmed medical diagnosis.

                    Do not use this result to make treatment
                    decisions. Consult a qualified healthcare
                    professional for medical advice.
                    """
                )

            except Exception as e:

                st.error("Disease prediction failed.")
                st.exception(e)


# ============================================================
# 19. HOW THE SYSTEM WORKS
# ============================================================

st.divider()
st.header("⚙️ How the System Works")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.info(
        """
        **1. Input**

        Patient enters symptoms or clinical description.
        """
    )

with col2:

    st.info(
        """
        **2. Preprocessing**

        Text is cleaned, tokenized and lemmatized.
        """
    )

with col3:

    st.info(
        """
        **3. TF-IDF**

        Text is converted into numerical feature vectors.
        """
    )

with col4:

    st.info(
        """
        **4. SVM**

        Linear Support Vector Machine predicts a disease category.
        """
    )


# ============================================================
# 20. ABOUT THE PROJECT
# ============================================================

st.divider()
st.header("ℹ️ About the Project")

st.write(
    """
    The Smart Healthcare Diagnosis Assistant is an educational
    Natural Language Processing and Machine Learning project.

    The system uses the Description column from DataSet_balanced.csv
    as its input feature and Disease_Name as its target variable.

    Text preprocessing includes lowercase conversion, URL
    removal, punctuation and digit removal, tokenization,
    stopword removal and lemmatization.

    TF-IDF converts the processed text into numerical features.

    A Linear Support Vector Machine classifier is used to
    predict the disease category. A predefined mapping then
    displays a suggested medical department for that disease.

    The trained model is loaded from a saved machine-learning
    pipeline created by model_train.py.
    """
)


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Healthcare Diagnosis Assistant | "
    "DataSet.csv | TF-IDF + LinearSVC"
)
