# Computer_Vision_4_Feature_Extraction_and_Matching_v1.5

- Source PDF: `ComputerScience/4-1_computer-vision/Computer_Vision_4_Feature_Extraction_and_Matching_v1.5.pdf`
- Total pages: 21

## Page 1

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현

Homography(호모그래피)

## Page 2

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현

𝐇

## Page 3

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현

𝐇

a b 𝐇a = b
𝐇c ≠ d

c d

## Page 4

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
ℎ11 ℎ21 ℎ31
𝐇AB = ℎ12 ℎ22 ℎ32
ℎ13 ℎ23 ℎ33

𝐇ABa1 = b1
𝐇ABa2 = b2

Homography(호모그래피)

## Page 5

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
ℎ11 ℎ21 ℎ31
𝐇AB = ℎ12 ℎ22 ℎ32
ℎ13 ℎ23 ℎ33
a1 = (xa1, ya1, 1)T, b1 = (xb1, yb1, 1)T 이라고 하면
ℎ11 ℎ21 ℎ31 xa1
𝐇ABa1 = ℎ12 ℎ22 ℎ32 ya1
ℎ13 ℎ23 ℎ33 1
ℎ11xa1 + ℎ21ya1 + ℎ31
= ℎ12xa1 + ℎ22ya1 + ℎ32 = b1
ℎ13xa1 + ℎ23ya1 + ℎ33

Homography(호모그래피)

## Page 6

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
ℎ11 ℎ21 ℎ31
𝐇AB = ℎ12 ℎ22 ℎ32
ℎ13 ℎ23 ℎ33
a1 = (xa1, ya1, 1)T, b1 = (xb1, yb1, 1)T 이라고 하면
ℎ11 ℎ21 ℎ31 xa1
𝐇ABa1 = ℎ12 ℎ22 ℎ32 ya1
ℎ13 ℎ23 ℎ33 1
ℎ11xa1 + ℎ21ya1 + ℎ31
= ℎ12xa1 + ℎ22ya1 + ℎ32 = b1
ℎ13xa1 + ℎ23ya1 + ℎ33
xa1 ya1
xa1 ya1
= xa1 ya1 = b1 (homogeneous coordinates)
Homography(호모그래피) xa1 ya1
1

## Page 7

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
ℎ11 ℎ21 ℎ31
𝐇AB = ℎ12 ℎ22 ℎ32
ℎ13 ℎ23 ℎ33
a1 = (xa1, ya1, 1)T, b1 = (xb1, yb1, 1)T 이라고 하면
ℎ11 ℎ21 ℎ31 xa1
𝐇ABa1 = ℎ12 ℎ22 ℎ32 ya1
ℎ13 ℎ23 ℎ33 1
xa1 ya1
xa1 ya1
= xa1 ya1 = b1 (homogeneous coordinates)
xa1 ya1
1
xa1 ya1 xa1 ya1
xb1= xa1 ya1 yb1= xa1 ya1
Homography(호모그래피)

## Page 8

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
- Homography를 결정하는 최소한의 매칭점들(matching points, correspondences)의 수는 4개임

𝐇 a1 = (xa1, ya1, 1)T, b1 = (xb1, yb1, 1)T 이라고 하면
a1
b1 = Ha1
- 한 ||b1 - Ha1|| = 0 a2 b1 b2
즉 ||b1 - Ha1|| = 0

b4
b1 × Ha1 = 0
a4 a3 b3

## Page 9

Homography(호모그래피) 추정
- 벡터(Vector)
- 벡터의 외적(cross product)
a2 b3 − a3 b2
- a×b = a2 b3 − a3 b2
a2 b3 − a3 b2

- ∣a×b∣=∣a∣∣b∣sinθ

a×a = 0

## Page 10

Homography(호모그래피) 추정
- Homography(호모그래피) 구하기0
a1 = (xa1, ya1, 1)T, b1 = (xb1, yb1, 1)T 이라고 하면

b1 × Ha1 = 0
xb1 ℎ11xa1 + ℎ21ya1 + ℎ31 a 2 b 3 − a3 b 2
yb1 × ℎ12xa1 + ℎ22ya1 + ℎ32 = 0 a×b = a2 b3 − a3 b2
1 ℎ13xa1 + ℎ23ya1 + ℎ33 ℎ11 a 2 b 3 − a3 b 2
ℎ12
ℎ13
0 0 0 −xa1 −ya1 −1 xa1yb1 ya1yb1 yb1 ℎ21 = 0 : 하나의 매칭점으로 2개의 방정식 얻음
xa1 ya1 1 0 0 0 −xa1xb1 −ya1xb1 −xb1 ℎ22 0
ℎ23
ℎ31
ℎ32
A h

## Page 11

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
- Homography를 결정하는 최소한의 매칭점들(matching points, correspondences)의 수는 4개임

𝐇
a1
a2 일치점 4개를 안다면
- 한 b1 b2
b1 = Ha1
b2 = Ha1 8개의 방정식
b3 = Ha1
b4 b4 = Ha1
a4 a3 b3 Ah = 0

## Page 12

Homography(호모그래피) 추정
- Homography(호모그래피)의 개념
- 평면형 물체(planar object)를 카메라로 찍었을 때 원래의 물체와 영상 평면에 투영된 물체 사이에 성립
되는 기하학적 변환 관계
- homogeneous coordinates에서 3x3 행렬 H로 표현
- Homography를 결정하는 최소한의 매칭점들(matching points, correspondences)의 수는 4개임

𝐇
a1
a2 일치점 4개를 안다면
- 한 b1 b2
b1 = Ha1
b2 = Ha1 8개의 방정식
b3 = Ha1
b4 b4 = Ha1 ℎ11 ℎ21 ℎ31
a3 b3 ℎ12 ℎ22 ℎ32
a4 Ah = 0
ℎ13 ℎ23 ℎ33
SVD(least squares)
8개의 변수 결정
( ℎ33 는 scale factor)

## Page 13

Homography(호모그래피) 추정
- Least squares(최소자승법)
- 오차의 제곱합을 최소화하여 최적의 해를 찾는 방법

- 아래와 같은 명확한 해가 존재

- 수학적으로 미분/최적화가 편리함
- Outlier에 매우 취약
- 오차2 의 합들이라서 큰 오차(outlier)가 결과에 큰
영향을 미침

## Page 14

Homography(호모그래피) 추정
- Least squares(최소자승법)
- 오차의 제곱합을 최소화하여 최적의 해를 찾는 방법

- 아래와 같은 명확한 해가 존재

- 수학적으로 미분/최적화가 편리함
- Outlier에 매우 취약
- 오차2 의 합들이라서 큰 오차(outlier)가 결과에 큰
영향을 미침

## Page 15

RANSAC(RANdom Sample Consensus)

- 소수의 깨끗한 샘플만 무작위로 골라 가설을 세우고, 가장 많은 지지를 받는 가설을 선택하는
알고리즘
- 전체 순서
1. 가설 설정 (Hypothesis): 데이터 중 아주 적은 수의 샘플(예: 직선의 경우 2개, homography의 경우 매칭
점 4개)을 무작위로 뽑음
2. 모델 생성: 뽑은 샘플들로 임시 모델을 계산
3. 검증 (Verification): 나머지 모든 데이터가 이 모델과 얼마나 가까운지 확인하고 정해진 오차 범위 내에
들어오면 인라이어(Inlier, 올바른 데이터)로 인정
4. 최적 모델 선택: 위 과정을 여러 번 반복하여, 가장 많은 인라이어(Inlier)를 확보한 모델을 최종 정답으
로 채택

## Page 16

RANSAC(RANdom Sample Consensus)

- Homography 추정을 위한 RANSAC
입력 : 매칭 쌍 집합 X={(a1, b1), (a2, b2),…, an, bn), }

X에서 네 쌍을 랜덤하게 선택하고 homography H 계산

X의 모든 매칭 쌍에 대해 아래와 같이 inlier 여부 판별
||bi - Hai|| < t 이면 inlier 아니면 outlier

출력 : 최적 homography Hopt

## Page 17

RANSAC(RANdom Sample Consensus)

- RANSAC 알고리즘의 특징
- 이상치(Outliers)에 매우 강함 (Robustness)
- Least Squares은 모든 데이터의 오차 제곱합을 최소화하려다 보니 단 하나의 아주 큰 노이즈만 있어도 모
델이 노이즈 쪽으로 크게 휘어버릴 수 있음
- RANSAC은 전체 중 아주 일부만 무작위로 뽑아서 모델을 만든 뒤 얼마나 많은 데이터가 이 모델에 동의
하는지(Consensus) 판단하기 때문에 데이터의 절반 이상이 Outliers여도 진짜 모델을 찾아낼 수 있음
- 특정 수학 모델에 국한되지 않고 다양한 기하학적 모델링에 모두 적용 가능
- 예) 직선은 2개의 점, 평면은 3개의 점, homography는 4개의 매칭점 등 모델을 구성하는 최소한의 샘플
을 선택할 수 있으면 적용 가능
- 샘플을 적게 뽑을수록 그 안에 오염된 데이터(Outlier)가 섞여 들어갈 확률이 적어지기 때문에 성
공 확률과 연산 효율이 좋아짐
- 예) 직선을 모델링할 때 3개의 점을 뽑을 때보다 2개의 점을 뽑을 때 outlier가 섞일 확률이 더 적음

## Page 18

RANSAC 실습
import cv2 des1_f32 = des1.astype('float32')
import numpy as np des2_f32 = des2.astype('float32')
import faiss
import urllib.request print(f"특징점 개수: Image1({len(kp1)}개), Image2({len(kp2)}개)\n")
from google.colab.patches import cv2_imshow
# 3. Faiss-GPU를 이용한 고속 KNN 매칭 (K=2)
# 1. GPU 리소스 초기화 및 이미지 로드 index = faiss.IndexFlatL2(128)
res = faiss.StandardGpuResources() gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(des2_f32)
# 서로 다른 시점에서 촬영된 이미지 두 장 distances, indices = gpu_index.search(des1_f32, 2)
url1 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg'
url2 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg' # 4. Lowe's Ratio Test를 통한 1차 필터링
urllib.request.urlretrieve(url1, 'aloeL.jpg') good_matches = []
urllib.request.urlretrieve(url2, 'aloeR.jpg') for i in range(len(des1)):
if distances[i][0] < 0.5 * distances[i][1]:
img1_gray = cv2.imread('aloeL.jpg', cv2.IMREAD_GRAYSCALE) m = cv2.DMatch(_queryIdx=i, _trainIdx=int(indices[i][0]),
_distance=distances[i][0])
img2_gray = cv2.imread('aloeR.jpg', cv2.IMREAD_GRAYSCALE)
good_matches.append(m)

# 2. SIFT 특징점 추출 (반복 패턴이라 특징점 개수를 충분히 늘림)
print(f"Faiss 매칭 완료 | Ratio Test 통과: {len(good_matches)}개")
sift = cv2.SIFT_create(nfeatures=5000)
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

## Page 19

RANSAC 실습
# 5. RANSAC + Homography 계산 및 최종 시각화 ㅁㅁ
if len(good_matches) > 4:
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC으로 homography 행렬(M) 계산
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
matches_mask = mask.ravel().tolist()

# Inliers만 추출
inlier_matches = [good_matches[i] for i in range(len(matches_mask)) if
matches_mask[i] == 1]
print(f"RANSAC 검증 완료 | 최종 인라이어(Inliers): {len(inlier_matches)}개")

res_img = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, inlier_matches[:500],
None,
flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("\n[RANSAC 매칭 결과]")
cv2_imshow(res_img)
else:
print("매칭점이 부족하여 호모그래피를 계산할 수 없습니다.")
}

## Page 20

내용 정리

- 특징 검출 및 기술 알고리즘(SIFT, ORB)
- 매칭 방법 및 관련 library
- 매칭의 성능 측정 지표

## Page 21

다음 주 강의 내용

- 스테레오 영상의 기하 구조
- 스테레오 영상으로부터 깊이 추정 방법

## Page 22

(텍스트 없음)
