
# ============================================================
# SMART HEALTHCARE DIAGNOSIS ASSISTANT
# MODEL TRAINING AND DISEASE PREDICTION
# TF-IDF + SUPPORT VECTOR MACHINE
# ============================================================

import os
import re
import string
import pickle
import warnings

import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "DataSet_balanced.csv")


# ============================================================
# 2. DATASET COLUMN NAMES
# ============================================================

TEXT_COLUMN = "Description"
TARGET_COLUMN = "Disease_Name"


# ============================================================
# 3. NLTK RESOURCE DOWNLOAD
# ============================================================

print("=" * 70)
print("SMART HEALTHCARE DIAGNOSIS ASSISTANT")
print("=" * 70)

print("\nChecking NLTK resources...")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("NLTK resources downloaded successfully.")


# ============================================================
# 4. NLTK OBJECTS
# ============================================================

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


# ============================================================
# 5. TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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
# DISEASE AND SUGGESTED DEPARTMENT MAPPING
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
    """Return a suggested department for a predicted disease."""
    return DISEASE_DEPARTMENT.get(
        str(disease).strip(),
        "General Medicine (Initial Consultation)"
    )


# ============================================================
# 6. CHECK DATASET FILE
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset file not found.")

    print("\nExpected dataset location:")
    print(DATASET_PATH)

    print(
        "\nPlace DataSet.csv in the same folder "
        "as model_train.py."
    )

    raise SystemExit


# ============================================================
# 7. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 9. CHECK REQUIRED COLUMNS
# ============================================================

if TEXT_COLUMN not in df.columns:

    print(f"\nERROR: Column '{TEXT_COLUMN}' not found.")

    print("Available columns:")
    print(df.columns.tolist())

    raise SystemExit


if TARGET_COLUMN not in df.columns:

    print(f"\nERROR: Column '{TARGET_COLUMN}' not found.")

    print("Available columns:")
    print(df.columns.tolist())

    raise SystemExit


print("\nRequired columns found successfully.")

print("Input feature:", TEXT_COLUMN)
print("Target variable:", TARGET_COLUMN)


# ============================================================
# 10. SELECT REQUIRED COLUMNS
# ============================================================

data = df[
    [
        TEXT_COLUMN,
        TARGET_COLUMN
    ]
].copy()


# ============================================================
# 11. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

print(data.isnull().sum())

# Remove records with missing input or target
data.dropna(
    subset=[
        TEXT_COLUMN,
        TARGET_COLUMN
    ],
    inplace=True
)


# ============================================================
# 12. CHECK DUPLICATE RECORDS
# ============================================================

# DataSet_balanced.csv may contain repeated rows created by
# oversampling. Keep these rows so that all disease classes
# retain the same number of records.
duplicate_count = data.duplicated().sum()

print("\nRepeated records found:", duplicate_count)
print("Repeated rows are retained to preserve class balance.")


# ============================================================
# 13. CLEAN EMPTY VALUES
# ============================================================

data[TEXT_COLUMN] = (
    data[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

data[TARGET_COLUMN] = (
    data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
)

data = data[
    (data[TEXT_COLUMN] != "") &
    (data[TARGET_COLUMN] != "")
].copy()

if data.empty:

    print("\nERROR: No valid records found in dataset.")
    raise SystemExit


# ============================================================
# 14. DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DISEASE DISTRIBUTION")
print("=" * 70)

disease_counts = data[TARGET_COLUMN].value_counts()

print("\nTotal unique diseases:")
print(data[TARGET_COLUMN].nunique())

print("\nDisease counts:")
print(disease_counts)


# ============================================================
# 15. TEXT PREPROCESSING EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("TEXT PREPROCESSING EXAMPLE")
print("=" * 70)

original_text = data[TEXT_COLUMN].iloc[0]

processed_text = preprocess_text(original_text)

print("\nOriginal text:")
print(original_text)

print("\nProcessed text:")
print(processed_text)


# ============================================================
# 16. APPLY TEXT PREPROCESSING
# ============================================================

print("\nPreprocessing dataset text...")

data["processed_text"] = (
    data[TEXT_COLUMN].apply(preprocess_text)
)

# Remove empty processed descriptions
data = data[
    data["processed_text"].str.strip() != ""
].copy()

print("Text preprocessing completed.")

if data.empty:

    print("\nERROR: No usable text remains after preprocessing.")
    raise SystemExit


# ============================================================
# 17. PREPARE INPUT AND TARGET
# ============================================================

X = data["processed_text"]

y = data[TARGET_COLUMN]

print("\n" + "=" * 70)
print("MODEL DATA")
print("=" * 70)

print("\nTotal records:", len(X))

print("Unique diseases:", y.nunique())

if y.nunique() < 2:

    print(
        "\nERROR: At least two different disease classes "
        "are required to train the SVM classifier."
    )

    raise SystemExit


# ============================================================
# 18. CHECK CLASS DISTRIBUTION
# ============================================================

class_counts = y.value_counts()

print("\nDisease classes with their record counts:")
print(class_counts)

can_evaluate = (
    len(class_counts) > 1
    and class_counts.min() >= 2
)


# ============================================================
# 19. CREATE TF-IDF + SVM PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("CREATING SVM PIPELINE")
print("=" * 70)

svm_pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
    ),

    (
        "svm",
        LinearSVC(
            class_weight="balanced",
            C=1.0,
            random_state=42
        )
    )

])

print("\nPipeline created successfully.")

print("Text -> TF-IDF -> Support Vector Machine")


# ============================================================
# 20. MODEL TRAINING AND EVALUATION
# ============================================================

accuracy = None
precision = None
recall = None
f1 = None

report = None
cm = None

if can_evaluate:

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Train model
    print("\nTraining Support Vector Machine...")

    svm_pipeline.fit(
        X_train,
        y_train
    )

    print("SVM training completed.")

    # Predictions
    y_pred = svm_pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("SVM MODEL PERFORMANCE")
    print("=" * 70)

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    # Confusion matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    evaluation_labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=evaluation_labels
    )

    print(cm)

    evaluation_type = "Stratified 75/25 test-set evaluation"

else:

    print("\nWARNING: Not enough examples per disease")
    print("for a reliable stratified test evaluation.")

    print("\nTraining using the complete dataset...")

    svm_pipeline.fit(X, y)

    print("SVM training completed.")

    evaluation_type = (
        "Full-dataset training; no held-out test metrics"
    )


# ============================================================
# 21. SAVE TRAINED MODEL
# ============================================================

pipeline_filename = os.path.join(
    BASE_DIR,
    "healthcare_svm_pipeline.pkl"
)

with open(pipeline_filename, "wb") as file:

    pickle.dump(
        svm_pipeline,
        file
    )

print("\nSaved model:", pipeline_filename)


# ============================================================
# 22. SAVE MODEL METRICS
# ============================================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "evaluation_type": evaluation_type,

    "total_records": len(data),

    "unique_diseases": y.nunique(),

    "input_column": TEXT_COLUMN,

    "target_column": TARGET_COLUMN

}

metrics_filename = os.path.join(
    BASE_DIR,
    "metrics.pkl"
)

with open(metrics_filename, "wb") as file:

    pickle.dump(metrics, file)

print("Saved metrics:", metrics_filename)


# ============================================================
# 23. SAVE CLASS LABELS
# ============================================================

class_labels = list(svm_pipeline.classes_)

labels_filename = os.path.join(
    BASE_DIR,
    "class_labels.pkl"
)

with open(labels_filename, "wb") as file:

    pickle.dump(class_labels, file)

print("Saved class labels:", labels_filename)


# ============================================================
# 24. SAVE DATASET INFORMATION
# ============================================================

dataset_info = {

    "dataset_shape": df.shape,

    "input_column": TEXT_COLUMN,

    "target_column": TARGET_COLUMN,

    "total_training_records": len(data),

    "number_of_diseases": y.nunique(),

    "dataset_name": os.path.basename(DATASET_PATH)

}

info_filename = os.path.join(
    BASE_DIR,
    "dataset_info.pkl"
)

with open(info_filename, "wb") as file:

    pickle.dump(dataset_info, file)

print("Saved dataset information:", info_filename)


# ============================================================
# 25. FINAL TRAINING OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nProject: Smart Healthcare Diagnosis Assistant")

print("\nDataset:", os.path.basename(DATASET_PATH))

print("Input feature:", TEXT_COLUMN)

print("Target variable:", TARGET_COLUMN)

print("Algorithm: Support Vector Machine")

print("Feature extraction: TF-IDF")

print("\nGenerated files:")

print("1. healthcare_svm_pipeline.pkl")
print("2. metrics.pkl")
print("3. class_labels.pkl")
print("4. dataset_info.pkl")

print("\nModel architecture:")
print("Text -> Preprocessing -> TF-IDF -> LinearSVC")

print("=" * 70)


# ============================================================
# 26. PREDICT DISEASE USING PATIENT SYMPTOMS
# ============================================================

print("\n" + "=" * 70)
print("SMART HEALTHCARE DISEASE PREDICTION")
print("=" * 70)

print("\nEnter patient symptoms to predict the disease.")
print("Type 'exit' to stop the prediction system.")

while True:

    # Get symptoms from user
    user_input = input(
        "\nEnter Patient Symptoms: "
    ).strip()

    # Exit condition
    if user_input.lower() == "exit":

        print("\nPrediction system closed.")
        break

    # Check empty input
    if not user_input:

        print("\nPlease enter patient symptoms.")
        continue

    # Preprocess entered symptoms
    processed_input = preprocess_text(user_input)

    if not processed_input:

        print("\nPlease enter valid symptoms.")
        continue

    try:

        # Predict disease using trained SVM model
        prediction = svm_pipeline.predict(
            [processed_input]
        )

        predicted_disease = str(prediction[0])

        # Find suggested medical department
        suggested_department = get_suggested_department(
            predicted_disease
        )

        # Display prediction result
        print("\n" + "-" * 70)
        print("          SMART HEALTHCARE DIAGNOSIS RESULT")
        print("-" * 70)

        print("\nEntered Symptoms:")
        print(user_input)

        print("\nProcessed Symptoms:")
        print(processed_input)

        print("\nPredicted Disease:")
        print(">>>", predicted_disease)

        print("\nSuggested Medical Department:")
        print(">>>", suggested_department)

        print("-" * 70)

        print(
            "\nNote: The department is a general educational suggestion. "
            "This model output is not a medical diagnosis or confirmed referral. "
            "Consult a qualified healthcare professional."
        )

    except Exception as e:

        print("\nPrediction Error:", e)


print(
    "\nThank you for using Smart Healthcare Diagnosis Assistant."
)
