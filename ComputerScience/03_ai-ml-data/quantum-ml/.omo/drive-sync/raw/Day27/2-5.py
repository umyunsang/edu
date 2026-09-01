"""
============================================================
Lab 15. QSVM Performance Evaluation


Accuracy 계산 및 분석
------------------------------------------------------------
이번 실습에서는

1. Prediction 결과 불러오기
2. Accuracy 계산
3. Accuracy 직접 계산
4. 두 결과 비교
5. Accuracy 결과 저장

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

from sklearn.metrics import accuracy_score

print("Library Import Completed")

# ============================================================
# STEP 2. Load Dataset
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Prediction Result")
print("=" * 70)

y_test = np.load("output/y_test.npy")
y_pred = np.load("output/y_prediction.npy")

print("Test Samples :", len(y_test))

# ============================================================
# STEP 3. Accuracy (Scikit-Learn)
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Accuracy (Scikit-Learn)")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")

# ============================================================
# STEP 4. Manual Accuracy
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Manual Accuracy")
print("=" * 70)

correct = np.sum(y_test == y_pred)

total = len(y_test)

manual_accuracy = correct / total

print("Correct Prediction :", correct)
print("Total Samples      :", total)

print(f"Manual Accuracy : {manual_accuracy:.4f}")

# ============================================================
# STEP 5. Compare Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Compare Accuracy")
print("=" * 70)

difference = abs(accuracy - manual_accuracy)

print(f"Scikit Accuracy : {accuracy:.6f}")
print(f"Manual Accuracy : {manual_accuracy:.6f}")
print(f"Difference      : {difference:.10f}")

if difference < 1e-10:
    print("Result : SAME")
else:
    print("Result : DIFFERENT")

# ============================================================
# STEP 6. Accuracy Report
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Accuracy Report")
print("=" * 70)

accuracy_df = pd.DataFrame({

    "Metric": [
        "Accuracy"
    ],

    "Value": [
        accuracy
    ],

    "Percentage": [
        accuracy * 100
    ]

})

print(accuracy_df)

# ============================================================
# STEP 7. Save Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Save Accuracy")
print("=" * 70)

accuracy_df.to_csv(

    "output/qsvm_accuracy.csv",

    index=False

)

print("Saved")

print("output/qsvm_accuracy.csv")

# ============================================================
# STEP 8. Save Accuracy Information
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Save Accuracy Info")
print("=" * 70)

accuracy_info = {

    "accuracy": accuracy,

    "correct_prediction": int(correct),

    "wrong_prediction": int(total - correct),

    "total_samples": int(total)

}

with open(

    "output/accuracy_info.pkl",

    "wb"

) as f:

    pickle.dump(

        accuracy_info,

        f

    )

print("output/accuracy_info.pkl")

# ============================================================
# STEP 9. Interpretation
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Accuracy Interpretation")
print("=" * 70)

if accuracy >= 0.95:

    level = "Excellent"

elif accuracy >= 0.90:

    level = "Very Good"

elif accuracy >= 0.80:

    level = "Good"

elif accuracy >= 0.70:

    level = "Fair"

else:

    level = "Needs Improvement"

print("Performance Level :", level)

print()

print("※ Accuracy는 전체적인 예측 정확도를 나타냅니다.")
print("※ Accuracy 하나만으로 모델을 평가해서는 안 됩니다.")
print("※ 다음 실습에서 Precision / Recall / F1 Score를 함께 분석합니다.")

# ============================================================
# STEP 10. Summary
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY SUMMARY")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Percentage      : {accuracy * 100:.2f}%")
print(f"Correct         : {correct}")
print(f"Wrong           : {total - correct}")
print(f"Performance     : {level}")

print("=" * 70)
print("Accuracy Analysis Completed")
print("Next Step : 2-6.py")
print("=" * 70)