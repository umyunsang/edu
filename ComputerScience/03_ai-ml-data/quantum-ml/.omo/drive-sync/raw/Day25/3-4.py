# ============================================================
# 학습된 QSVM을 이용한 Prediction 수행
# ============================================================

import time
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


# ============================================================
# STEP 1. Lab 3 결과 확인
# ============================================================

print("=" * 80)
print("STEP 1. Iris Dataset 불러오기")
print("=" * 80)

iris = load_iris()

X = iris.data
y = iris.target

print(f"전체 Sample 수 : {X.shape[0]}")
print(f"Feature 수     : {X.shape[1]}")
print(f"Class 수       : {len(np.unique(y))}")

print("\nFeature 이름")
print(iris.feature_names)

print("\nClass 이름")
print(iris.target_names)


# ============================================================
# STEP 2. Feature 선택
# Petal Length, Petal Width 사용
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Feature 선택")
print("=" * 80)

# Petal Length, Petal Width
X = X[:, 2:4]

print("선택된 Feature")
print("- Petal Length")
print("- Petal Width")

print(f"\n변경된 Feature Shape : {X.shape}")


# ============================================================
# STEP 3. Binary Classification 구성
# Setosa(0), Versicolor(1)만 사용
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Binary Classification 구성")
print("=" * 80)

mask = y != 2

X = X[mask]
y = y[mask]

print(f"Sample 수 : {X.shape[0]}")
print(f"Class : {np.unique(y)}")

print("\nClass Distribution")

unique, counts = np.unique(y, return_counts=True)

for c, n in zip(unique, counts):
    print(f"Class {c} : {n}")


# ============================================================
# STEP 4. Train / Test Split
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Train / Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")


# ============================================================
# STEP 5. Feature Scaling
# Quantum Feature Map 입력 범위 : 0 ~ π
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Feature Scaling")
print("=" * 80)

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling 완료")

print("\nTrain Data")

print("Minimum")

print(X_train_scaled.min(axis=0))

print("Maximum")

print(X_train_scaled.max(axis=0))


# ============================================================
# STEP 6. 데이터 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. 전처리 데이터 확인")
print("=" * 80)

print(f"X_train_scaled : {X_train_scaled.shape}")

print(f"X_test_scaled  : {X_test_scaled.shape}")

print(f"y_train        : {y_train.shape}")

print(f"y_test         : {y_test.shape}")

print("\nTraining Sample (첫 5개)")

df_train = pd.DataFrame(
    X_train_scaled,
    columns=[
        "Petal Length",
        "Petal Width"
    ]
)

print(df_train.head())

print("\nTraining Label")

print(y_train[:5])


print("=" * 80)
print("STEP 1. Lab 1 결과 확인")
print("=" * 80)

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")

feature_dimension = X_train_scaled.shape[1]

print()
print(f"Feature Dimension : {feature_dimension}")



# ============================================================
# QSVM 학습 시작
# ============================================================


# ============================================================
# STEP 2. Quantum Feature Map 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Quantum Feature Map 생성")
print("=" * 80)

feature_map = zz_feature_map(
    feature_dimension=feature_dimension,
    reps=2,
    entanglement="linear"
)

print(feature_map)

print()

print("Qubit 수 :", feature_map.num_qubits)
print("Parameter 수 :", feature_map.num_parameters)


# ============================================================
# STEP 3. Quantum Feature Map 회로 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Feature Map 회로")
print("=" * 80)

print(
    feature_map.draw(output="text")
)

# matplotlib 환경이라면
# feature_map.draw(output="mpl")


# ============================================================
# STEP 4. Quantum Kernel 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Quantum Kernel 생성")
print("=" * 80)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print(type(quantum_kernel))

print()

print(
    "Feature Map Qubits :",
    quantum_kernel.feature_map.num_qubits
)


# ============================================================
# STEP 5. Quantum Kernel 확인
# (교육용 Sample만 사용)
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Sample Kernel Matrix")
print("=" * 80)

sample = X_train_scaled[:3]

kernel_matrix = quantum_kernel.evaluate(
    x_vec=sample
)

print(kernel_matrix)

print()

print("Kernel Matrix Shape :", kernel_matrix.shape)


# ============================================================
# STEP 6. QSVC 객체 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. QSVC 객체 생성")
print("=" * 80)

qsvc = QSVC(
    quantum_kernel=quantum_kernel
)

print(qsvc)

print()

print(type(qsvc))



print("=" * 80)
print("STEP 1. Lab 2 결과 확인")
print("=" * 80)

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"y_train        : {y_train.shape}")

print()

print("Feature Dimension :", X_train_scaled.shape[1])

print("Class :", np.unique(y_train))

print()

print("QSVC Type")

print(type(qsvc))


# ============================================================
# STEP 2. Training Data 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Training Data")
print("=" * 80)

print("Training Sample (First 5)")

print(X_train_scaled[:5])

print()

print("Training Label")

print(y_train[:10])


# ============================================================
# STEP 3. 학습 전 QSVC 상태
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. QSVC 상태 확인")
print("=" * 80)

print(qsvc)

print()


# ============================================================
# STEP 4. QSVM 학습
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. QSVM 학습")
print("=" * 80)

print("Training Start")

start_time = time.time()

qsvc.fit(
    X_train_scaled,
    y_train
)

end_time = time.time()

training_time = end_time - start_time

print()

print("Training Complete")

print(f"Training Time : {training_time:.4f} sec")


# ============================================================
# STEP 5. 학습된 Class 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Learned Classes")
print("=" * 80)

print("Classes")

print(qsvc.classes_)


# ============================================================
# STEP 6. Support Vector 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. Support Vector")
print("=" * 80)

print("Support Vector Index")

print(qsvc.support_)

print()

print("Support Vector 개수")

print(qsvc.n_support_)

print()

print(
    "Total Support Vector :",
    len(qsvc.support_)
)


# ============================================================
# STEP 7. Decision Function 정보
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. Decision Function")
print("=" * 80)

print("Intercept")

print(qsvc.intercept_)

print()

print("Dual Coefficient Shape")

print(qsvc.dual_coef_.shape)


# ============================================================
# STEP 8. 학습 결과 저장
# ============================================================

training_result = {

    "training_time": training_time,

    "classes": qsvc.classes_,

    "support_vector": qsvc.support_,

    "support_vector_count": qsvc.n_support_,

    "model": qsvc

}



print("=" * 80)
print("STEP 1. Lab 3 결과 확인")
print("=" * 80)

print("학습 완료 모델")

print(type(qsvc))

print()

print("Test Data Shape")

print(X_test_scaled.shape)

print()

print("Test Label Shape")

print(y_test.shape)


# ============================================================
# 학습된 QSVM을 이용한 Prediction 수행 시작
# ============================================================



# ============================================================
# STEP 2. Prediction 수행
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Prediction 수행")
print("=" * 80)

y_pred = qsvc.predict(
    X_test_scaled
)

print("Prediction 완료")


# ============================================================
# STEP 3. Prediction 결과 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Prediction 결과")
print("=" * 80)

print(y_pred)


# ============================================================
# STEP 4. True Label 비교
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. True Label 비교")
print("=" * 80)

print("True Label")

print(y_test)

print()

print("Prediction")

print(y_pred)


# ============================================================
# STEP 5. Sample별 Prediction 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Sample별 Prediction")
print("=" * 80)

for i in range(len(y_test)):

    print(f"Sample {i+1}")

    print("True       :", y_test[i])

    print("Prediction :", y_pred[i])

    print("-" * 40)


# ============================================================
# STEP 6. 정분류 여부 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. Prediction 확인")
print("=" * 80)

for i, (true, pred) in enumerate(zip(y_test, y_pred), start=1):

    if true == pred:
        result = "Correct"
    else:
        result = "Incorrect"

    print(
        f"Sample {i:2d} : {result}"
    )


# ============================================================
# STEP 7. Prediction 결과 표 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. Prediction Table")
print("=" * 80)

result_df = pd.DataFrame({

    "True Label": y_test,

    "Prediction": y_pred

})

print(result_df)


# ============================================================
# STEP 8. Class 이름으로 출력
# ============================================================

print("\n" + "=" * 80)
print("STEP 8. Class 이름 출력")
print("=" * 80)

class_names = [

    "Setosa",

    "Versicolor"

]

for i, (true, pred) in enumerate(zip(y_test, y_pred), start=1):

    print(

        f"Sample {i:2d}"

    )

    print(

        "True :",

        class_names[true]

    )

    print(

        "Prediction :",

        class_names[pred]

    )

    print("-" * 40)


# ============================================================
# STEP 9. 새로운 데이터 Prediction
# ============================================================

print("\n" + "=" * 80)
print("STEP 9. 새로운 데이터 Prediction")
print("=" * 80)

new_sample = [

    [1.4, 0.2]

]

new_sample_scaled = scaler.transform(
    new_sample
)

new_prediction = qsvc.predict(
    new_sample_scaled
)

print("입력 데이터")

print(new_sample)

print()

print("Prediction")

print(

    class_names[
        new_prediction[0]
    ]

)


# ============================================================
# STEP 10. Prediction 결과 저장
# ============================================================

prediction_result = {

    "prediction": y_pred,

    "true_label": y_test,

    "result_table": result_df

}


# ============================================================
# STEP 11. 실습 완료
# ============================================================

print("\n" + "=" * 80)
print("Lab 4 완료")
print("=" * 80)

print("Prediction이 완료되었습니다.")

print()

print("생성된 결과")

print("- y_pred")

print("- Prediction Table")

print("- True Label 비교")

print()

