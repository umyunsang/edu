"""
============================================================
Lab 15. QSVM Performance Evaluation


Classification Report 생성 및 분석
------------------------------------------------------------
이번 실습에서는

1. Prediction 결과 불러오기
2. Classification Report 출력
3. Dictionary 변환
4. DataFrame 생성
5. CSV 저장
6. Class별 성능 분석
7. Macro / Weighted Average 비교

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

from sklearn.metrics import classification_report

print("Library Import Completed")

# ============================================================
# STEP 2. Load Prediction
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Prediction")
print("=" * 70)

y_test = np.load("output/y_test.npy")
y_pred = np.load("output/y_prediction.npy")

print("Samples :", len(y_test))

# ============================================================
# STEP 3. Classification Report
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Classification Report")
print("=" * 70)

report_text = classification_report(
    y_test,
    y_pred
)

print(report_text)

# ============================================================
# STEP 4. Dictionary Report
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Dictionary Report")
print("=" * 70)

report_dict = classification_report(

    y_test,

    y_pred,

    output_dict=True

)

print(report_dict.keys())

# ============================================================
# STEP 5. DataFrame
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. DataFrame")
print("=" * 70)

report_df = pd.DataFrame(report_dict).transpose()

print(report_df)

# ============================================================
# STEP 6. Save CSV
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Save CSV")
print("=" * 70)

report_df.to_csv(

    "output/qsvm_classification_report.csv"

)

print("Saved")

print("output/qsvm_classification_report.csv")

# ============================================================
# STEP 7. Save Pickle
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Save Pickle")
print("=" * 70)

with open(

    "output/classification_report.pkl",

    "wb"

) as f:

    pickle.dump(

        report_dict,

        f

    )

print("output/classification_report.pkl")

# ============================================================
# STEP 8. Class Performance
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Class Performance")
print("=" * 70)

labels = [

    key

    for key in report_dict.keys()

    if key not in [

        "accuracy",

        "macro avg",

        "weighted avg"

    ]

]

for label in labels:

    print(f"\nClass : {label}")

    print(

        f"Precision : {report_dict[label]['precision']:.4f}"

    )

    print(

        f"Recall    : {report_dict[label]['recall']:.4f}"

    )

    print(

        f"F1 Score  : {report_dict[label]['f1-score']:.4f}"

    )

    print(

        f"Support   : {int(report_dict[label]['support'])}"

    )

# ============================================================
# STEP 9. Accuracy / Macro / Weighted
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Overall Performance")
print("=" * 70)

print(f"Accuracy : {report_dict['accuracy']:.4f}")

print()

print("Macro Average")

print(

    report_df.loc["macro avg"]

)

print()

print("Weighted Average")

print(

    report_df.loc["weighted avg"]

)

# ============================================================
# STEP 10. Interpretation
# ============================================================

print("\n" + "=" * 70)
print("STEP 10. Interpretation")
print("=" * 70)

macro = report_dict["macro avg"]["f1-score"]

weighted = report_dict["weighted avg"]["f1-score"]

difference = abs(

    macro -

    weighted

)

print(f"Macro F1     : {macro:.4f}")

print(f"Weighted F1  : {weighted:.4f}")

print(f"Difference   : {difference:.4f}")

print()

if difference < 0.02:

    print("클래스별 성능이 비교적 균형적입니다.")

else:

    print("클래스 불균형의 영향을 받을 가능성이 있습니다.")

print()

print("※ Accuracy는 전체 성능입니다.")

print("※ Macro Average는 모든 클래스를 동일하게 평가합니다.")

print("※ Weighted Average는 Support를 고려합니다.")

# ============================================================
# STEP 11. Summary
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT SUMMARY")
print("=" * 70)

print(report_df)

print()

print("=" * 70)
print("Classification Report Completed")
print("Next Step : 2-8.py")
print("=" * 70)