# ============================================================
# SMART HEALTHCARE DIAGNOSIS ASSISTANT
# HEALTHCARE TEXT DATA ANALYSIS AND MACHINE LEARNING
# DATASET: DataSet.csv
# ALGORITHM: SUPPORT VECTOR MACHINE (LinearSVC)
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import re
import string
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk

from collections import Counter
from wordcloud import WordCloud

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

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
# 2. PROJECT DIRECTORY AND DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "DataSet_balanced.csv"
)

TEXT_COLUMN = "Description"
TARGET_COLUMN = "Disease_Name"


# ============================================================
# 3. DOWNLOAD NLTK RESOURCES
# ============================================================

print("\nChecking NLTK resources...")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("NLTK resources downloaded successfully.")


# ============================================================
# 4. INITIALIZE NLTK OBJECTS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# 5. PROJECT HEADER
# ============================================================

print("\n" + "=" * 70)

print("SMART HEALTHCARE DIAGNOSIS ASSISTANT")

print("=" * 70)


# ============================================================
# 6. CHECK DATASET FILE
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset file not found.")

    print("\nExpected location:")
    print(DATASET_PATH)

    raise SystemExit(
        "Please place DataSet_balanced.csv in the project folder."
    )


# ============================================================
# 7. LOAD DATASET
# ============================================================

print("\nLoading healthcare dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 8. DISPLAY FIRST FIVE RECORDS
# ============================================================

print("\n" + "=" * 70)

print("FIRST FIVE RECORDS")

print("=" * 70)

print(df.head())


# ============================================================
# 9. DATASET SHAPE
# ============================================================

print("\nDATASET SHAPE")

print("Rows:", df.shape[0])

print("Columns:", df.shape[1])


# ============================================================
# 10. DISPLAY COLUMN NAMES
# ============================================================

print("\nDATASET COLUMNS")

for i, column in enumerate(df.columns, start=1):

    print(i, ".", column)


# ============================================================
# 11. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)

print("DATASET INFORMATION")

print("=" * 70)

df.info()


# ============================================================
# 12. BASIC STATISTICS
# ============================================================

print("\nBASIC STATISTICS")

print(
    df.describe(include="all")
)


# ============================================================
# 13. CHECK REQUIRED COLUMNS
# ============================================================

print("\nCHECKING REQUIRED COLUMNS")

if TEXT_COLUMN not in df.columns:

    print("Input column not found:", TEXT_COLUMN)

    print("Available columns:", df.columns.tolist())

    raise SystemExit

if TARGET_COLUMN not in df.columns:

    print("Target column not found:", TARGET_COLUMN)

    print("Available columns:", df.columns.tolist())

    raise SystemExit

print("Input feature:", TEXT_COLUMN)

print("Target variable:", TARGET_COLUMN)


# ============================================================
# 14. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)

print("MISSING VALUE ANALYSIS")

print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing values in each column:")

print(missing_values)


missing_percentage = (
    df.isnull().sum() / len(df)
) * 100


missing_summary = pd.DataFrame({

    "Missing Values": missing_values,

    "Percentage": missing_percentage

})


print("\nMISSING VALUE SUMMARY")

print(missing_summary)


# ============================================================
# 15. DUPLICATE RECORD ANALYSIS
# ============================================================

print("\n" + "=" * 70)

print("DUPLICATE RECORD ANALYSIS")

print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    "Number of duplicate records:",
    duplicate_count
)


# ============================================================
# 16. SELECT INPUT AND TARGET COLUMNS
# ============================================================

print("\n" + "=" * 70)

print("SELECTED DATASET COLUMNS")

print("=" * 70)

df_model = df[
    [
        TEXT_COLUMN,
        TARGET_COLUMN
    ]
].copy()

print("Input feature:", TEXT_COLUMN)

print("Target variable:", TARGET_COLUMN)

print("\nSample records:")

print(df_model.head())


# ============================================================
# 17. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)

print("DATA CLEANING")

print("=" * 70)

before_cleaning = len(df_model)

df_model.dropna(
    subset=[
        TEXT_COLUMN,
        TARGET_COLUMN
    ],
    inplace=True
)

print(
    "Records removed due to missing values:",
    before_cleaning - len(df_model)
)


# ============================================================
# 18. REMOVE DUPLICATE RECORDS
# ============================================================

before_duplicates = len(df_model)

df_model.drop_duplicates(
    inplace=True
)

after_duplicates = len(df_model)

print(
    "Duplicate records removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 19. CLEAN EMPTY VALUES
# ============================================================

df_model[TEXT_COLUMN] = (
    df_model[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

df_model[TARGET_COLUMN] = (
    df_model[TARGET_COLUMN]
    .astype(str)
    .str.strip()
)

df_model = df_model[
    (df_model[TEXT_COLUMN] != "") &
    (df_model[TARGET_COLUMN] != "")
].copy()

print(
    "\nDataset shape after cleaning:",
    df_model.shape
)


# ============================================================
# 20. DISEASE CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("DISEASE CLASS DISTRIBUTION")

print("=" * 70)

class_distribution = (
    df_model[TARGET_COLUMN]
    .value_counts()
)

print(class_distribution)

print(
    "\nTotal number of disease categories:",
    df_model[TARGET_COLUMN].nunique()
)


# ============================================================
# 21. VISUALIZE DISEASE DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 8))

class_distribution.plot(
    kind="barh"
)

plt.title(
    "Disease Category Distribution"
)

plt.xlabel(
    "Number of Records"
)

plt.ylabel(
    "Disease Name"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
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

    return text


# ============================================================
# 23. APPLY TEXT CLEANING
# ============================================================

print("\n" + "=" * 70)

print("TEXT CLEANING")

print("=" * 70)

df_model["clean_text"] = (
    df_model[TEXT_COLUMN]
    .apply(clean_text)
)

print("\nOriginal text:")

print(
    df_model[TEXT_COLUMN].iloc[0]
)

print("\nCleaned text:")

print(
    df_model["clean_text"].iloc[0]
)


# ============================================================
# 24. TOKENIZATION
# ============================================================

print("\n" + "=" * 70)

print("TOKENIZATION")

print("=" * 70)

df_model["tokens"] = (
    df_model["clean_text"]
    .apply(word_tokenize)
)

print("\nTokenized text:")

print(
    df_model["tokens"].iloc[0]
)


# ============================================================
# 25. STOPWORD REMOVAL
# ============================================================

print("\n" + "=" * 70)

print("STOPWORD REMOVAL")

print("=" * 70)


def remove_stopwords(tokens):

    filtered_tokens = [

        word

        for word in tokens

        if word not in stop_words

    ]

    return filtered_tokens


df_model["tokens_no_stopwords"] = (
    df_model["tokens"]
    .apply(remove_stopwords)
)

print("\nText after stopword removal:")

print(
    df_model["tokens_no_stopwords"].iloc[0]
)


# ============================================================
# 26. LEMMATIZATION
# ============================================================

print("\n" + "=" * 70)

print("LEMMATIZATION")

print("=" * 70)


def lemmatize_tokens(tokens):

    return [

        lemmatizer.lemmatize(word)

        for word in tokens

    ]


df_model["lemmatized_tokens"] = (
    df_model["tokens_no_stopwords"]
    .apply(lemmatize_tokens)
)


# Convert tokens back to text
df_model["processed_text"] = (
    df_model["lemmatized_tokens"]
    .apply(
        lambda tokens: " ".join(tokens)
    )
)


print("\nText after lemmatization:")

print(
    df_model["processed_text"].iloc[0]
)


# ============================================================
# 27. REMOVE EMPTY PROCESSED TEXT
# ============================================================

df_model = df_model[
    df_model["processed_text"].str.strip() != ""
].copy()

print(
    "\nRecords after text preprocessing:",
    len(df_model)
)


# ============================================================
# 28. WORD FREQUENCY ANALYSIS
# ============================================================

print("\n" + "=" * 70)

print("WORD FREQUENCY ANALYSIS")

print("=" * 70)

all_text = " ".join(
    df_model["processed_text"]
)

all_words = all_text.split()

word_frequency = Counter(
    all_words
)

top_words = word_frequency.most_common(20)

frequency_df = pd.DataFrame(
    top_words,
    columns=[
        "Word",
        "Frequency"
    ]
)

print("\nTOP 20 FREQUENT WORDS")

print(frequency_df)


# ============================================================
# 29. BAR CHART OF FREQUENT WORDS
# ============================================================

plt.figure(figsize=(12, 7))

plt.barh(
    frequency_df["Word"],
    frequency_df["Frequency"]
)

plt.title(
    "Top 20 Most Frequent Healthcare Words"
)

plt.xlabel(
    "Frequency"
)

plt.ylabel(
    "Word"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 30. WORD CLOUD
# ============================================================

print("\nGenerating word cloud...")

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(all_text)


plt.figure(figsize=(15, 8))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Healthcare Text Word Cloud"
)

plt.show()


# ============================================================
# 31. PREPARE MODEL INPUT AND TARGET
# ============================================================

print("\n" + "=" * 70)

print("PREPARING MODEL DATA")

print("=" * 70)

X = df_model["processed_text"]

y = df_model[TARGET_COLUMN]

print(
    "Input records:",
    len(X)
)

print(
    "Disease categories:",
    y.nunique()
)


# ============================================================
# 32. CHECK CLASS DISTRIBUTION FOR SPLITTING
# ============================================================

class_counts = y.value_counts()

if len(class_counts) < 2:

    raise SystemExit(
        "At least two disease categories are required."
    )

if class_counts.min() < 2:

    raise SystemExit(
        "Each disease category needs at least two records "
        "for stratified train-test evaluation. "
        "Please check the dataset."
    )


# ============================================================
# 33. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)

print("TRAIN-TEST SPLIT")

print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 34. DEFINE THREE MACHINE LEARNING MODELS
# ============================================================

print("\n" + "=" * 70)
print("DEFINING SVM, LOGISTIC REGRESSION AND NAIVE BAYES")
print("=" * 70)

models = {
    "SVM": LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Naive Bayes": MultinomialNB()
}

results = []
trained_models = {}
predictions = {}


# ============================================================
# 35. TRAIN AND EVALUATE ALL THREE MODELS
# ============================================================

for name, classifier in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    model_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
        ),
        ("classifier", classifier)
    ])

    # Train the model using the same training data.
    model_pipeline.fit(X_train, y_train)

    # Predict disease labels for the same test data.
    y_pred = model_pipeline.predict(X_test)

    trained_models[name] = model_pipeline
    predictions[name] = y_pred

    # Calculate weighted evaluation metrics.
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

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    })

    print(name, "training and evaluation completed.")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")


# ============================================================
# 36. DISPLAY MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

print(results_df.round(2).to_string(index=False))


# ============================================================
# 37. FIND BEST MODEL USING WEIGHTED F1 SCORE
# ============================================================

best_index = results_df["F1 Score"].idxmax()
best_model_name = results_df.loc[best_index, "Algorithm"]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print("\n" + "=" * 70)
print("BEST PERFORMING MODEL")
print("=" * 70)
print("Best model based on weighted F1 Score:", best_model_name)
print(
    "Weighted F1 Score:",
    f"{results_df.loc[best_index, 'F1 Score']:.2f}%"
)


# ============================================================
# 38. CLASSIFICATION REPORT FOR EACH MODEL
# ============================================================

for name in models:
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT -", name)
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions[name],
            zero_division=0
        )
    )


# ============================================================
# 39. CONFUSION MATRIX FOR THE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX -", best_model_name)
print("=" * 70)

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels
)

print(cm)


# ============================================================
# 40. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 11))

plt.imshow(
    cm,
    interpolation="nearest",
    aspect="auto"
)

plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")

plt.xticks(
    ticks=np.arange(len(labels)),
    labels=labels,
    rotation=90
)

plt.yticks(
    ticks=np.arange(len(labels)),
    labels=labels
)

plt.colorbar()
plt.tight_layout()
plt.show()


# ============================================================
# 41. ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison of Machine Learning Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# ============================================================
# 42. COMPARISON GRAPHS FOR ALL METRICS
# ============================================================

metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for metric in metrics_to_plot:

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Algorithm"],
        results_df[metric]
    )

    plt.title(metric + " Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel(metric + " (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


# ============================================================
# 43. TF-IDF FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE INFORMATION")
print("=" * 70)

tfidf_model = best_model.named_steps["tfidf"]

print(
    "Number of TF-IDF features:",
    len(tfidf_model.get_feature_names_out())
)

print(
    "N-gram range:",
    tfidf_model.ngram_range
)


# ============================================================
# 44. SAVE THE BEST TRAINED MODEL
# ============================================================

print("\nSaving the best trained model...")

model_path = os.path.join(
    BASE_DIR,
    "best_healthcare_model.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

print("Best model saved:", model_path)


# ============================================================
# 45. SAVE MODEL COMPARISON RESULTS
# ============================================================

comparison_path = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("Model comparison saved:", comparison_path)


# ============================================================
# 46. SAVE BEST MODEL METRICS
# ============================================================

best_metrics = results_df.loc[best_index].to_dict()

metrics = {
    "best_model": best_model_name,
    "accuracy": best_metrics["Accuracy"] / 100,
    "precision": best_metrics["Precision"] / 100,
    "recall": best_metrics["Recall"] / 100,
    "f1_score": best_metrics["F1 Score"] / 100,
    "evaluation_type": "Stratified 75/25 test-set evaluation",
    "total_records": len(df_model),
    "unique_diseases": y.nunique(),
    "input_column": TEXT_COLUMN,
    "target_column": TARGET_COLUMN,
    "all_model_results": results
}

metrics_path = os.path.join(
    BASE_DIR,
    "metrics.pkl"
)

with open(metrics_path, "wb") as file:
    pickle.dump(metrics, file)

print("Best model metrics saved:", metrics_path)


# ============================================================
# 47. SAVE CLASS LABELS
# ============================================================

class_labels = list(best_model.classes_)

labels_path = os.path.join(
    BASE_DIR,
    "class_labels.pkl"
)

with open(labels_path, "wb") as file:
    pickle.dump(class_labels, file)

print("Class labels saved:", labels_path)


# ============================================================
# 48. SAVE DATASET INFORMATION
# ============================================================

dataset_info = {
    "dataset_shape": df.shape,
    "input_column": TEXT_COLUMN,
    "target_column": TARGET_COLUMN,
    "total_training_records": len(df_model),
    "number_of_diseases": y.nunique(),
    "dataset_name": os.path.basename(DATASET_PATH),
    "algorithms_compared": list(models.keys()),
    "selected_best_model": best_model_name
}

info_path = os.path.join(
    BASE_DIR,
    "dataset_info.pkl"
)

with open(info_path, "wb") as file:
    pickle.dump(dataset_info, file)

print("Dataset information saved:", info_path)


# ============================================================
# 49. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SMART HEALTHCARE DIAGNOSIS ASSISTANT")
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPROJECT DETAILS")
print("Dataset:", os.path.basename(DATASET_PATH))
print("Input Feature:", TEXT_COLUMN)
print("Target Variable:", TARGET_COLUMN)
print("Feature Extraction: TF-IDF")
print("N-grams: Unigrams and Bigrams")
print("Algorithms Compared: SVM, Logistic Regression, Naive Bayes")

print("\nMODEL PERFORMANCE")
print(results_df.round(2).to_string(index=False))

print("\nBEST MODEL:", best_model_name)

print("\nGENERATED FILES")
print("1. best_healthcare_model.pkl")
print("2. metrics.pkl")
print("3. class_labels.pkl")
print("4. dataset_info.pkl")
print("5. model_comparison.csv")

print("\n" + "=" * 70)
print("PROJECT EXECUTION FINISHED")
print("=" * 70)
