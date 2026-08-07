"""
============================================================
Lab QSVM Performance Evaluation


Precision / Recall / F1 Score 계산
------------------------------------------------------------
이번 실습에서는

1. Prediction 결과 불러오기
2. Precision 계산
3. Recall 계산
4. F1 Score 계산
5. Macro Average 계산
6. Weighted Average 계산
7. Accuracy와 비교
8. 결과 저장

============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Library Import Completed")

# ============================================================
# STEP 2. Load Prediction Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Prediction")
print("=" * 70)

y_test = np.load("output/y_test.npy")
y_pred = np.load("output/y_prediction.npy")

print("Samples :", len(y_test))

# ============================================================
# STEP 3. Accuracy
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Accuracy")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

# ============================================================
# STEP 4. Precision
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Precision")
print("=" * 70)

precision = precision_score(
    y_test,
    y_pred
)

print(f"Precision : {precision:.4f}")

# ============================================================
# STEP 5. Recall
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Recall")
print("=" * 70)

recall = recall_score(
    y_test,
    y_pred
)

print(f"Recall : {recall:.4f}")

# ============================================================
# STEP 6. F1 Score
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. F1 Score")
print("=" * 70)

f1 = f1_score(
    y_test,
    y_pred
)

print(f"F1 Score : {f1:.4f}")

# ============================================================
# STEP 7. Macro Average
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Macro Average")
print("=" * 70)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1 Score  : {macro_f1:.4f}")

# ============================================================
# STEP 8. Weighted Average
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Weighted Average")
print("=" * 70)

weighted_precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

weighted_recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"Weighted Precision : {weighted_precision:.4f}")
print(f"Weighted Recall    : {weighted_recall:.4f}")
print(f"Weighted F1 Score  : {weighted_f1:.4f}")

# ============================================================
# STEP 9. Summary DataFrame
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Summary")
print("=" * 70)

result_df = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1
    ]

})

print(result_df)

# ============================================================
# STEP 10. Save Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 10. Save Result")
print("=" * 70)

result_df.to_csv(
    "output/qsvm_metrics.csv",
    index=False
)

print("Saved")

print("output/qsvm_metrics.csv")

# ============================================================
# STEP 11. Save Metric Information
# ============================================================

print("\n" + "=" * 70)
print("STEP 11. Save Metric Info")
print("=" * 70)

metric_info = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "macro_f1": macro_f1,

    "weighted_f1": weighted_f1

}

with open(
    "output/metric_info.pkl",
    "wb"
) as f:

    pickle.dump(
        metric_info,
        f
    )

print("output/metric_info.pkl")

# ============================================================
# STEP 12. Interpretation
# ============================================================

print("\n" + "=" * 70)
print("STEP 12. Interpretation")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print()

if abs(precision-recall) < 0.03:

    print("Precision과 Recall이 균형을 이루는 모델입니다.")

elif precision > recall:

    print("Precision이 Recall보다 높습니다.")
    print("Positive 예측은 정확하지만 일부 Positive를 놓치고 있습니다.")

else:

    print("Recall이 Precision보다 높습니다.")
    print("Positive를 잘 찾지만 False Positive가 증가할 수 있습니다.")

print()

print("※ Accuracy 하나만으로 모델을 평가해서는 안 됩니다.")
print("※ 다음 실습에서는 Classification Report를 생성합니다.")

# ============================================================
# STEP 13. Summary
# ============================================================

print("\n" + "=" * 70)
print("METRIC SUMMARY")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")

print("=" * 70)
print("Metric Analysis Completed")
print("Next Step : 2-7.py")
print("=" * 70)