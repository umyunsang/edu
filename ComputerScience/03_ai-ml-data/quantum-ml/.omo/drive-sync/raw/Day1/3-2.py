import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 데이터 생성
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,     # 실제 분류에 사용되는 특성
    n_redundant=0,       # 중복 특성 제거
    n_repeated=0,
    n_classes=2,
    random_state=42
)

print("X Shape :", X.shape)
print("y Shape :", y.shape)

# 시각화
plt.figure(figsize=(6, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Binary Classification Dataset")
plt.colorbar(label="Class")
plt.show()