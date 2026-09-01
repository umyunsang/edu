import os
import pickle

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report

)

print("="*70)
print("STEP 2. Load Dataset")
print("="*70)

OUTPUT_DIR = "output"

CLASSICAL_DIR = os.path.join(
    OUTPUT_DIR,
    "classical"
)

os.makedirs(CLASSICAL_DIR, exist_ok=True)

X_train = np.load(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    )
)

X_test = np.load(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    )
)

y_train = np.load(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    )
)

y_test = np.load(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    )
)

print("Dataset Loaded")

print("="*70)
print("STEP 4. Classical SVM")
print("="*70)

svm = SVC(
    kernel="rbf",
    random_state=42
)

svm.fit(
    X_train,
    y_train
)

print("Training Complete")

prediction = svm.predict(X_test)

score = svm.decision_function(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction
)

recall = recall_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)

cm = confusion_matrix(
    y_test,
    prediction
)

report = classification_report(
    y_test,
    prediction
)

metric_info = {

    "accuracy":accuracy,

    "precision":precision,

    "recall":recall,

    "f1":f1

}

prediction_info = {

    "y_test":y_test,

    "y_prediction":prediction,

    "y_score":score

}

with open(

    os.path.join(

        CLASSICAL_DIR,

        "metric_info.pkl"

    ),

    "wb"

) as f:

    pickle.dump(metric_info,f)

with open(

    os.path.join(

        CLASSICAL_DIR,

        "classification_report.pkl"

    ),

    "wb"

) as f:

    pickle.dump(report,f)

with open(

    os.path.join(

        CLASSICAL_DIR,

        "confusion_matrix.pkl"

    ),

    "wb"

) as f:

    pickle.dump(cm,f)

with open(

    os.path.join(

        CLASSICAL_DIR,

        "prediction_info.pkl"

    ),

    "wb"

) as f:

    pickle.dump(prediction_info,f)

with open(

    os.path.join(

        CLASSICAL_DIR,

        "performance_report.txt"

    ),

    "w"

) as f:

    f.write("Classical SVM Performance Report\n")

    f.write("="*50+"\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")

    f.write(f"Precision : {precision:.4f}\n")

    f.write(f"Recall : {recall:.4f}\n")

    f.write(f"F1 Score : {f1:.4f}\n\n")

    f.write(report)

print("="*70)
print("CLASSICAL BASELINE COMPLETE")
print("="*70)

print(f"Accuracy : {accuracy:.4f}")

print()

print("Saved")

print("- metric_info.pkl")

print("- prediction_info.pkl")

print("- confusion_matrix.pkl")

print("- classification_report.pkl")

print("- performance_report.txt")





