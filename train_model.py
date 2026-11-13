import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Load dataset
data = pd.read_csv(r"C:\Users\Keert\Downloads\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv")


data = data.drop(columns=[
    "UDI", "Product ID",
    "TWF", "HDF", "PWF", "OSF", "RNF"
])

# Separate features and target
X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

# Convert categorical column
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nPredicted Values:")
print(pd.Series(y_pred).value_counts())

print("\nF1 Score:", f1_score(y_test, y_pred))

# Save model
joblib.dump(model, "machine_failure_model.pkl")


joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")