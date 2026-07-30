"""
============================================================
Lab 15. QSVM Performance Evaluation


QSVM Performance Report
------------------------------------------------------------
이번 실습에서는

1. 모든 평가 결과 불러오기
2. 성능 요약
3. Confusion Matrix 분석
4. ROC / AUC 분석
5. 최종 Performance Report 생성

============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import os
import pickle

import pandas as pd

print("Library Import Completed")

# ============================================================
# STEP 2. Load Accuracy
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Accuracy")
print("=" * 70)

accuracy_df = pd.read_csv(
    "output/qsvm_accuracy.csv"
)

accuracy = accuracy_df.loc[
    0,
    "Value"
]

print(f"Accuracy : {accuracy:.4f}")

# ============================================================
# STEP 3. Load Metrics
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Load Metrics")
print("=" * 70)

metric_df = pd.read_csv(
    "output/qsvm_metrics.csv"
)

print(metric_df)

precision = metric_df.loc[
    metric_df["Metric"]=="Precision",
    "Value"
].values[0]

recall = metric_df.loc[
    metric_df["Metric"]=="Recall",
    "Value"
].values[0]

f1 = metric_df.loc[
    metric_df["Metric"]=="F1 Score",
    "Value"
].values[0]

# ============================================================
# STEP 4. Load Classification Report
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Classification Report")
print("=" * 70)

class_df = pd.read_csv(
    "output/qsvm_classification_report.csv",
    index_col=0
)

print(class_df)

# ============================================================
# STEP 5. Load Confusion Matrix
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Confusion Matrix")
print("=" * 70)

cm_df = pd.read_csv(
    "output/confusion_matrix.csv",
    index_col=0
)

print(cm_df)

with open(
    "output/confusion_matrix.pkl",
    "rb"
) as f:

    cm = pickle.load(f)

TN = cm["TN"]
FP = cm["FP"]
FN = cm["FN"]
TP = cm["TP"]

# ============================================================
# STEP 6. Load ROC
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. ROC / AUC")
print("=" * 70)

with open(
    "output/roc_info.pkl",
    "rb"
) as f:

    roc = pickle.load(f)

auc = roc["auc"]

print(f"AUC : {auc:.4f}")

# ============================================================
# STEP 7. Summary Table
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Summary Table")
print("=" * 70)

summary = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]

})

print(summary)

summary.to_csv(
    "output/performance_summary.csv",
    index=False
)

# ============================================================
# STEP 8. Final Evaluation
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Final Evaluation")
print("=" * 70)

score = 0

if accuracy >= 0.90:
    score += 1

if precision >= 0.90:
    score += 1

if recall >= 0.90:
    score += 1

if f1 >= 0.90:
    score += 1

if auc >= 0.90:
    score += 1

if score == 5:

    level = "Excellent"

elif score >= 4:

    level = "Very Good"

elif score >= 3:

    level = "Good"

elif score >= 2:

    level = "Fair"

else:

    level = "Needs Improvement"

print("Overall Performance :", level)

# ============================================================
# STEP 9. Improvement Guide
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Improvement Guide")
print("=" * 70)

if precision < recall:

    print("Precision 개선")

    print("- Kernel Parameter 조정")

    print("- Feature Map 변경")

elif recall < precision:

    print("Recall 개선")

    print("- Class Weight 조정")

    print("- Data Balance 개선")

else:

    print("Precision / Recall 균형")

print()

print("추가 개선 방법")

print("- Feature Map 변경")

print("- Quantum Kernel 변경")

print("- Train Data 증가")

print("- Hyperparameter Tuning")

# ============================================================
# STEP 10. Report
# ============================================================

print("\n" + "=" * 70)
print("STEP 10. Save Report")
print("=" * 70)

report = f"""

====================================================
QSVM PERFORMANCE REPORT
====================================================

Accuracy  : {accuracy:.4f}

Precision : {precision:.4f}

Recall    : {recall:.4f}

F1 Score  : {f1:.4f}

AUC        : {auc:.4f}

----------------------------------------------------

TN : {TN}

FP : {FP}

FN : {FN}

TP : {TP}

----------------------------------------------------

Overall Performance

{level}

====================================================

"""

with open(

    "output/performance_report.txt",

    "w"

) as f:

    f.write(report)

print(report)

print("Saved")

print("output/performance_report.txt")

# ============================================================
# STEP 11. Output Check
# ============================================================

print("\n" + "=" * 70)
print("STEP 11. Output Files")
print("=" * 70)

files = sorted(os.listdir("output"))

for file in files:

    print(file)

# ============================================================
# STEP 12. Summary
# ============================================================

print("\n" + "=" * 70)
print("LAB SUMMARY")
print("=" * 70)

print("Prediction")

print("   ↓")

print("Accuracy")

print("   ↓")

print("Precision / Recall / F1")

print("   ↓")

print("Classification Report")

print("   ↓")

print("Confusion Matrix")

print("   ↓")

print("ROC Curve")

print("   ↓")

print("Performance Report")

print()

print("Congratulations!")

print("QSVM Performance Evaluation Completed.")

print("=" * 70)