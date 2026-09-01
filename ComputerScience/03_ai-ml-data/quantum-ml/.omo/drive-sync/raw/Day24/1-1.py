# ============================================================
# Lab. Classical SVM Baseline
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ============================================================
# STEP 1. Iris Dataset
# ============================================================

print("=" * 80)
print("STEP 1. Load Iris Dataset")
print("=" * 80)

iris = load_iris()

X = iris.data
y = iris.target

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)

# ============================================================
# STEP 2. Binary Classification
# (Setosa vs Versicolor)
# ============================================================

print()
print("=" * 80)
print("STEP 2. Binary Classification")
print("=" * 80)

mask = y != 2

X = X[mask]
y = y[mask]

print("Binary Dataset :", X.shape)

# ============================================================
# STEP 3. Train/Test Split
# ============================================================

print()
print("=" * 80)
print("STEP 3. Train/Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train :", X_train.shape)
print("Test  :", X_test.shape)

# ============================================================
# STEP 4. Create Classical SVM
# ============================================================

print()
print("=" * 80)
print("STEP 4. Create SVM")
print("=" * 80)

svm = SVC()

print(svm)

# ============================================================
# STEP 5. Model Training
# ============================================================

print()
print("=" * 80)
print("STEP 5. Training")
print("=" * 80)

svm.fit(X_train, y_train)

print("Training Complete")

# ============================================================
# STEP 6. Prediction
# ============================================================

print()
print("=" * 80)
print("STEP 6. Prediction")
print("=" * 80)

prediction = svm.predict(X_test)

print("Prediction")

for pred, label in zip(prediction, y_test):
    print(f"Predict : {pred}    Label : {label}")

# ============================================================
# STEP 7. Accuracy
# ============================================================

print()
print("=" * 80)
print("STEP 7. Accuracy")
print("=" * 80)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy : {accuracy:.4f}")

# ============================================================
# STEP 8. Confusion Matrix
# ============================================================

print()
print("=" * 80)
print("STEP 8. Confusion Matrix")
print("=" * 80)

cm = confusion_matrix(y_test, prediction)

print(cm)

# ============================================================
# STEP 9. Classification Report
# ============================================================

print()
print("=" * 80)
print("STEP 9. Classification Report")
print("=" * 80)

print(classification_report(y_test, prediction))

# ============================================================
# STEP 10. Support Vector
# ============================================================

print()
print("=" * 80)
print("STEP 10. Support Vector")
print("=" * 80)

print("Number of Support Vectors")

print(svm.n_support_)

print()

print("Support Vector Index")

print(svm.support_)

print()

print("Support Vector")

print(svm.support_vectors_)

# ============================================================
# STEP 11. Summary
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")

print("This model will be used as the Baseline")
print("for comparison with QSVM.")
