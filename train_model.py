import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    )

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("hotel_bookings.csv")
print("Original dataset shape:", df.shape)

# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# ============================================================
# 3. REMOVE LEAKAGE / POST-OUTCOME COLUMNS
# ============================================================

columns_to_drop = [
    "company",
    "reservation_status",
    "reservation_status_date"
]

df = df.drop(columns=columns_to_drop)

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# ============================================================
# 7. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Gaussian Naive Bayes needs a dense matrix.
dense_categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

dense_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", dense_categorical_pipeline, categorical_features)
    ]
)

# ============================================================
# 8. MODELS COVERED IN THE PROJECT
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocessor", dense_preprocessor),
        ("model", GaussianNB())
    ]),

    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
    ]),

    "SVM": Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearSVC(
        random_state=42,
        max_iter=3000
    ))
]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "AdaBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Artificial Neural Network": Pipeline([
        ("preprocessor", preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=42
        ))
    ])
}

# ============================================================
# 9. TRAIN AND EVALUATE ALL MODELS
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# F1 Score is used as the primary selection criterion.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("SELECTED MODEL:", best_model_name)
print("=" * 60)

# ============================================================
# 11. SAVE OUTPUTS FOR STREAMLIT
# ============================================================

joblib.dump(best_model, "hotel_booking_model.pkl")
results_df.to_csv("model_comparison.csv", index=False)

with open("best_model_name.txt", "w", encoding="utf-8") as f:
    f.write(best_model_name)

print("\nSaved:")
print("- hotel_booking_model.pkl")
print("- model_comparison.csv")
print("- best_model_name.txt")
