# ==========================================
# Iris Classification Practice
# ==========================================

# 1. 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Step 1. 데이터 로드
# ==========================================

iris = load_iris()

print("=" * 50)
print("Iris Dataset Information")
print("=" * 50)

print("Feature Names")
print(iris.feature_names)

print("\nTarget Names")
print(iris.target_names)

# ==========================================
# Step 2. 데이터 확인
# ==========================================

print("\nFirst 5 Features")
print(iris.data[:5])

print("\nFirst 5 Labels")
print(iris.target[:5])

# ==========================================
# Step 3. Feature(X), Label(y) 분리
# ==========================================

X = iris.data
y = iris.target

print("\nFeature Shape :", X.shape)
print("Label Shape :", y.shape)

# ==========================================
# Step 4. Train/Test 분리
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Data :", X_train.shape)
print("Test Data :", X_test.shape)

# ==========================================
# Step 5. 모델 생성
# ==========================================

model = LogisticRegression(
    max_iter=300
)

# ==========================================
# Step 6. 모델 학습
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Complete")

# ==========================================
# Step 7. 예측 수행
# ==========================================

pred = model.predict(X_test)

print("\nPrediction Result")
print(pred)

# ==========================================
# Step 8. 정확도 평가
# ==========================================

accuracy = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# ==========================================
# Step 9. 실제값 vs 예측값 비교
# ==========================================

print("\nActual Label")
print(y_test)

print("\nPredicted Label")
print(pred)

# ==========================================
# Step 10. 새로운 데이터 예측
# ==========================================

new_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(new_flower)

print("\nNew Flower Prediction")
print("Predicted Label :", result[0])

print(
    "Flower Name :",
    iris.target_names[result[0]]
)

print("\nPractice Complete!")