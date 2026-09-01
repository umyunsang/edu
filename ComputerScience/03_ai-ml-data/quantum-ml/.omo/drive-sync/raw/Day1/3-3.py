import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# 20차원 데이터 생성
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,      # 실제 분류에 영향을 주는 Feature
    n_redundant=5,         # 중복 Feature
    n_repeated=0,
    n_classes=2,
    random_state=42
)

print("X Shape :", X.shape)
print("y Shape :", y.shape)

# PCA를 이용한 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA Shape :", X_pca.shape)

# 시각화
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="coolwarm",
    alpha=0.7,
    edgecolors='k'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("20-Dimensional Data Projected to 2D using PCA")
plt.colorbar(label="Class")
plt.show()