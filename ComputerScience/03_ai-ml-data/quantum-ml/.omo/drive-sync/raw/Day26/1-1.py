# ============================================================
# Part 5. Confusion Matrix 생성 및 평가 지표 계산
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 1. Actual Label 준비
# ============================================================

print("=" * 80)
print("STEP 1. Actual Label 준비")
print("=" * 80)

# 실제 정답(Label)

y_true = np.array([
    0,1,1,0,1,
    0,0,1,1,0,
    1,1,0,0,1,
    0,1,0,1,0
])

print("Actual Label")

print(y_true)

print()

# ============================================================
# STEP 2. Prediction 결과 준비
# ============================================================

print("=" * 80)
print("STEP 2. Prediction 결과 준비")
print("=" * 80)

# 모델의 예측 결과
# (실습용 예제)

y_pred = np.array([
    0,1,0,0,1,
    0,0,1,1,0,
    1,0,0,0,1,
    1,1,0,1,0
])

print("Prediction")

print(y_pred)

print()

# ============================================================
# STEP 3. Actual과 Prediction 비교
# ============================================================

print("=" * 80)
print("STEP 3. Actual과 Prediction 비교")
print("=" * 80)

result_df = pd.DataFrame({

    "Actual": y_true,
    "Prediction": y_pred

})

result_df["Result"] = np.where(
    result_df["Actual"] == result_df["Prediction"],
    "Correct",
    "Wrong"
)

print(result_df)

print()

correct = (result_df["Result"] == "Correct").sum()
wrong = (result_df["Result"] == "Wrong").sum()

print(f"Correct : {correct}")
print(f"Wrong   : {wrong}")

print()

# ============================================================
# STEP 4. Confusion Matrix 생성
# ============================================================

print("=" * 80)
print("STEP 4. Confusion Matrix 생성")
print("=" * 80)

cm = confusion_matrix(y_true, y_pred)

print(cm)

print()

# ============================================================
# STEP 5. TP, TN, FP, FN 추출
# ============================================================

print("=" * 80)
print("STEP 5. TP, TN, FP, FN 확인")
print("=" * 80)

tn, fp, fn, tp = cm.ravel()

print(f"True Positive  (TP) : {tp}")
print(f"True Negative  (TN) : {tn}")
print(f"False Positive (FP) : {fp}")
print(f"False Negative (FN) : {fn}")

print()

# ============================================================
# STEP 6. Confusion Matrix 시각화
# ============================================================

print("=" * 80)
print("STEP 6. Confusion Matrix 출력")
print("=" * 80)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative","Positive"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# STEP 7. Accuracy 계산
# ============================================================

print("=" * 80)
print("STEP 7. Accuracy 계산")
print("=" * 80)

accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy : {accuracy:.4f}")

print()

# ============================================================
# STEP 8. Precision 계산
# ============================================================

print("=" * 80)
print("STEP 8. Precision 계산")
print("=" * 80)

precision = precision_score(y_true, y_pred)

print(f"Precision : {precision:.4f}")

print()

# ============================================================
# STEP 9. Recall 계산
# ============================================================

print("=" * 80)
print("STEP 9. Recall 계산")
print("=" * 80)

recall = recall_score(y_true, y_pred)

print(f"Recall : {recall:.4f}")

print()

# ============================================================
# STEP 10. F1 Score 계산
# ============================================================

print("=" * 80)
print("STEP 10. F1 Score 계산")
print("=" * 80)

f1 = f1_score(y_true, y_pred)

print(f"F1 Score : {f1:.4f}")

print()

# ============================================================
# STEP 11. 성능 비교표 출력
# ============================================================

print("=" * 80)
print("STEP 11. Performance Report")
print("=" * 80)

performance = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1
    ]

})

performance["Value"] = performance["Value"].round(4)

print(performance)

print()

# ============================================================
# STEP 12. Classification Report
# ============================================================

print("=" * 80)
print("STEP 12. Classification Report")
print("=" * 80)

report = classification_report(y_true, y_pred)

print(report)

# ============================================================
# STEP 13. 결과 해석
# ============================================================

print("=" * 80)
print("STEP 13. 결과 해석")
print("=" * 80)

print(f"Accuracy  : {accuracy:.2%}")
print(f"Precision : {precision:.2%}")
print(f"Recall    : {recall:.2%}")
print(f"F1 Score  : {f1:.2%}")

print()

if recall < 0.7:

    print("Recall이 낮습니다.")
    print("→ 실제 Positive 데이터를 많이 놓치고 있습니다.")

elif precision < 0.7:

    print("Precision이 낮습니다.")
    print("→ False Positive가 많이 발생합니다.")

else:

    print("Precision과 Recall이 비교적 균형적입니다.")

print()

print("Confusion Matrix를 함께 확인하여")
print("FP와 FN 중 어떤 오류가 더 많은지 분석해 보세요.")

print()

# ============================================================
# STEP 14. 실습 종료
# ============================================================

print("=" * 80)
print("실습 종료")
print("=" * 80)