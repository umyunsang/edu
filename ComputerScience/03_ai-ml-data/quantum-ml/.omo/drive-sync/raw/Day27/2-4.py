"""
============================================================
Lab 15. QSVM Performance Evaluation


Prediction 수행
------------------------------------------------------------
이번 실습에서는

1. 학습 완료 모델 불러오기
2. Test Dataset 불러오기
3. Prediction 수행
4. Prediction Time 측정
5. Prediction DataFrame 생성
6. Prediction 결과 저장

============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import pickle
import time

import numpy as np
import pandas as pd

print("Library Import Completed")

# ============================================================
# STEP 2. Load Dataset
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Dataset")
print("=" * 70)

X_test = np.load("output/X_test.npy")
y_test = np.load("output/y_test.npy")

print("Test Shape :", X_test.shape)
print("Label Shape:", y_test.shape)

# ============================================================
# STEP 3. Load Trained Model
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Load QSVM Model")
print("=" * 70)

with open("output/qsvc_model.pkl", "rb") as f:
    qsvc = pickle.load(f)

print("QSVM Model Loaded")

# ============================================================
# STEP 4. Prediction
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Prediction")
print("=" * 70)

prediction_start = time.perf_counter()

y_pred = qsvc.predict(X_test)

# ROC/AUC 계산에 사용할 연속적인 Decision Score
y_score = qsvc.decision_function(X_test)

prediction_end = time.perf_counter()

prediction_time = prediction_end - prediction_start

print("Prediction Completed")
print(f"Prediction Time : {prediction_time:.6f} sec")

print("Prediction Shape :", y_pred.shape)
print("Score Shape      :", y_score.shape)

# ============================================================
# STEP 5. Prediction Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Prediction Result")
print("=" * 70)

print("Actual Labels")

print(y_test)

print()

print("Predicted Labels")

print(y_pred)

# ============================================================
# STEP 6. Prediction DataFrame
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Prediction DataFrame")
print("=" * 70)

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Prediction": y_pred,

    "Correct": y_test == y_pred

})

print(prediction_df)

# ============================================================
# STEP 7. Correct / Wrong Count
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Prediction Summary")
print("=" * 70)

correct_count = np.sum(y_test == y_pred)

wrong_count = np.sum(y_test != y_pred)

print("Correct Prediction :", correct_count)

print("Wrong Prediction   :", wrong_count)

# ============================================================
# STEP 8. Save Prediction Result
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Save Prediction")
print("=" * 70)

prediction_df.to_csv(

    "output/qsvm_predictions.csv",

    index=False

)

np.save(
    "output/y_prediction.npy",
    y_pred
)

np.save(
    "output/y_score.npy",
    y_score
)

print("Saved")
print("- output/qsvm_predictions.csv")
print("- output/y_prediction.npy")
print("- output/y_score.npy")

# ============================================================
# STEP 9. Save Prediction Information
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Save Prediction Info")
print("=" * 70)

prediction_info = {

    "prediction_time": prediction_time,

    "correct": int(correct_count),

    "wrong": int(wrong_count),

    "total": len(y_test)

}

with open(

    "output/prediction_info.pkl",

    "wb"

) as f:

    pickle.dump(

        prediction_info,

        f

    )

print("output/prediction_info.pkl")

# ============================================================
# STEP 10. Summary
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print("Test Samples        :", len(y_test))

print("Prediction Time     : %.6f sec" % prediction_time)

print("Correct Prediction  :", correct_count)

print("Wrong Prediction    :", wrong_count)

print("=" * 70)
print("Prediction Completed")
print("Next Step : 2-5.py")
print("=" * 70)