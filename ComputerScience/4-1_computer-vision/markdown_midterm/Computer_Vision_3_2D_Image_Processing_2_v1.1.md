# Computer_Vision_3_2D_Image_Processing_2_v1.1

- Source PDF: `ComputerScience/4-1_computer-vision/Computer_Vision_3_2D_Image_Processing_2_v1.1.pdf`
- Total pages: 26

## Page 1

코너 검출

- 코너 검출(Corner detection) 알고리즘

왼쪽 영상의 a, b, c 중 오른쪽 영상에서 가장 찾기 쉬운 것은?

## Page 2

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

(u, v) 방향의 영상의 밝기의 변화량

보통 -1 ≤ u ≤ 1, -1 ≤ v ≤ 1
w(x, y) : Gaussian 필터 또는 1
-1, -1

1, 1

## Page 3

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

(u, v) 방향의 영상의 밝기의 변화량

보통 -1 ≤ u ≤ 1, -1 ≤ v ≤ 1
w(x, y) : Gaussian 필터 또는 1
-1, -1

1, 1

E(u, v)
원래 영상

a가 코너로 가장 적당

## Page 4

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

where

Ix, Iy : x, y 방향에서의 밝기 변화(Sobel 에지값)

## Page 5

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

where

Ix, Iy : x, y 방향에서의 밝기 변화(Sobel 에지값)

Eigen value decomposition(고유값분해)

λ x, λ y

## Page 6

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

where

Ix, Iy : x, y 방향에서의 밝기 변화(Sobel 에지값)

Eigen value decomposition(고유값분해)

λ x, λ y k: 상수

## Page 7

코너 검출

- 해리스 코너 검출(Harris corner detection) 알고리즘

## Page 8

해리스 코너 검출 실습
import urllib.request # 검출된 코너 점들을 약간 크고 뚜렷하게 만듦
from google.colab.patches import cv2_imshow dst = cv2.dilate(dst, None)

# 샘플 이미지 다운로드 # 영상 내 가장 강한 코너 점의 값(dst.max())의 1%보다 큰 픽셀만 코너로 간주
url = threshold = 0.01 * dst.max()
'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.jpg'
urllib.request.urlretrieve(url, 'sudoku.jpg')
# dst > threshold 조건을 만족하는 좌표의 픽셀을 빨간색 [0, 0, 255]로 변경
img = cv2.imread('sudoku.jpg')
img_result = img.copy()
img_result[dst > threshold] = [0, 0, 255]
# 코너 검출은 그레이 이미지에서 수행
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 결과 출력
gray = np.float32(gray) # 계산 정밀도를 위해 float32로 변환
print("--- [원본 이미지] ---")
cv2_imshow(img)
# 해리스 코너 검출 수행
print("\n--- [해리스 코너 검출 결과 (빨간색 점)] ---")
# gray: 입력영상 (float32 타입)
cv2_imshow(img_result)
# 2: 블록 크기 (Block Size). 코너 검출을 위한 윈도우 크기
# 3: 소벨 커널 크기 (Sobel Kernel Size). 미분 값을 계산할 때 사용하는 커널 크기
# - 0.04: Harris 코너 검출 방정식의 k값 (보통 0.04~0.06 사용)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

## Page 9

슈퍼픽셀(Superpixel) 분할

- 슈퍼픽셀(Superpixel)은 픽셀보다 크지만 물체보다 작은 자잘한 영역으로 과잉 분할(over-
segmentation)된 단위
- SLIC(Simple Linear Iterative Clustering) 알고리즘
- 각 픽셀을 (R, G, B, x, y)의 5차원 데이터로 취급
- k-means clustering과 비슷하게 동작 (주변 픽셀 할당 단계와 cluster들의 중심을 갱신하는 단계를 반복)
- 파라미터
- K : 슈퍼픽셀의 개수
- Compactness : 얼마나 촘촘하게 뭉칠지 결정

SLIC 알고리즘의 초기 cluster들의 중심

## Page 10

슈퍼픽셀(Superpixel) 분할

- 슈퍼픽셀(Superpixel)은 픽셀보다 크지만 물체보다 작은 자잘한 영역으로 과잉 분할(over-
segmentation)된 단위
- SLIC(Simple Linear Iterative Clustering) 알고리즘
- 각 픽셀을 (R, G, B, x, y)의 5차원 데이터로 취급
- k-means clustering과 비슷하게 동작 (주변 픽셀 할당 단계와 cluster들의 중심을 갱신하는 단계를 반복)
- 파라미터
- K : 슈퍼픽셀의 개수
- Compactness : 얼마나 촘촘하게 뭉칠지 결정

## Page 11

슈퍼픽셀(Superpixel) 분할

- 슈퍼픽셀(Superpixel)은 픽셀보다 크지만 물체보다 작은 자잘한 영역으로 과잉 분할(over-
segmentation)된 단위
- SLIC(Simple Linear Iterative Clustering) 알고리즘
- 각 픽셀을 (R, G, B, x, y)의 5차원 데이터로 취급
- k-means clustering과 비슷하게 동작 (주변 픽셀 할당 단계와 cluster들의 중심을 갱신하는 단계를 반복)
- 파라미터
- K : 슈퍼픽셀의 개수
- Compactness : 얼마나 촘촘하게 뭉칠지 결정
- 장점
- 연산 효율성이 커짐. 예) 픽셀 수가 1,000,000개인 이미지를 500개의 슈퍼 픽셀로 줄이면 후속 알
고리즘의 계산량이 크게 감소
- 영상 내 실제 사물의 경계선을 어느 정도 보존하면서 뭉쳐줌
- 활용
- 주로 슈퍼 픽셀들로 그래프를 만들어서 활용(객체 분할, 깊이추정 등)

## Page 12

슈퍼픽셀(Superpixel) 분할 실습
import urllib.request # 결과 시각화
import matplotlib.pyplot as plt plt.figure(figsize=(18, 6))
from skimage.segmentation import slic, mark_boundaries plt.subplot(1, 3, 1)
from skimage.color import label2rgb plt.title("Original (Butterfly)")
plt.imshow(img_rgb)
# 샘플 이미지 다운로드 plt.axis('off')
url =
'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg'
plt.subplot(1, 3, 2)
urllib.request.urlretrieve(url, 'butterfly.jpg')
plt.title("SLIC Boundaries (400 segments)")
img = cv2.imread('butterfly.jpg')
plt.imshow(mark_boundaries(img_rgb, segments))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.axis('off')

# SLIC 알고리즘 적용
plt.subplot(1, 3, 3)
segments = slic(img_rgb, n_segments=400, compactness=10, sigma=1, start_label=1)
plt.title("Segmented (Avg Color)")
plt.imshow(superpixel_avg.astype('uint8'))
# 평균 색상으로 채우기
plt.axis('off')
superpixel_avg = label2rgb(segments, img_rgb, kind='avg')

plt.tight_layout()
plt.show()

## Page 13

최적화 분할

- 그래프의 표현
- 그래프는 현상이나 사물을 정점(vertex)과 간선(edge)로 표현
- 그래프 G=(V, E)
- V : n개의 정점 집합, E : 정점 간에 존재하는 간선 집합
- 두 정점이 간선으로 연결되어 있으면 인접(adjacent)하다고 함

가중치를 가진 무방향 그래프 가중치를 가진 방향 그래프

## Page 14

최적화 분할

- 영상의 그래프 표현
- 픽셀 또는 슈퍼픽셀을 노드(정점)로 취함
- 두 노드 vp, vq의 유사도를 아래 식으로 계산하여 간선에 부여
- f(v)는 v에 해당하는 픽셀의 색상(R, G, B)과 위치 (x, y)를 결합한 5차원 벡터
- v가 슈퍼 픽셀인 경우 슈퍼픽셀 내 픽셀들의 평균을 사용
D : 상수

## Page 15

최적화 분할

- 영상의 그래프 표현
- Normalized cut(N-cut, 정규화 절단)알고리즘
- cut은 영상을 두 영역으로 분할했을 때 분할의 좋은 정도를 측정해 주는 목적 함수
- C1과 C2가 클수록 둘 사이에 간선이 많아 cut은 덩달아 커지므로 cut을 사용한 분할 알고리즘
은 영역을 자잘하게 분할하는 경향이 있음

- N-cut은 cut을 정규화하여 영역의 크기에 중립이 되게 해줌

## Page 16

최적화 분할

- 영상의 그래프 표현
- Normalized cut(N-cut, 정규화 절단)알고리즘
- cut은 영상을 두 영역으로 분할했을 때 분할의 좋은 정도를 측정해 주는 목적 함수
- C1과 C2가 클수록 둘 사이에 간선이 많아 cut은 덩달아 커지므로 cut을 사용한 분할 알고리즘
은 영역을 자잘하게 분할하는 경향이 있음

- N-cut은 cut을 정규화하여 영역의 크기에 중립이 되게 해줌
cut

C1과 C가 얼마나 강하게
연결되어 있는가?

## Page 17

최적화 분할

- 영상의 그래프 표현
- Normalized cut(N-cut, 정규화 절단)알고리즘
- cut은 영상을 두 영역으로 분할했을 때 분할의 좋은 정도를 측정해 주는 목적 함수
- C1과 C2가 클수록 둘 사이에 간선이 많아 cut은 덩달아 커지므로 cut을 사용한 분할 알고리즘
은 영역을 자잘하게 분할하는 경향이 있음

- N-cut은 cut을 정규화하여 영역의 크기에 중립이 되게 해줌

- 한계
- 색상과 거리 정보에만 의존하므로 의미 분할 불가능

## Page 18

최적화 분할 실습
import urllib.request # 슈퍼픽셀들 간의 인접 관계와 색상 차이를 계산하여 그래프 생성
import matplotlib.pyplot as plt g = graph.rag_mean_color(img_rgb, labels)
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb # N-Cut (Normalized Cut) 적용
from skimage import graph # N-Cut 계산을 위한 그래프 모듈 # thresh: 잘라낼 기준값. 낮을수록 더 많이 쪼개지고, 높을수록 크게 뭉침
# num_cuts: 반복해서 자를 횟수
# 샘플 이미지 로드 nc_labels = graph.cut_normalized(labels, g, thresh=1.0, num_cuts=2)
url =
'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg'
# 결과 시각화 준비
urllib.request.urlretrieve(url, 'butterfly.jpg')
slic_avg = label2rgb(labels, img_rgb, kind='avg') # SLIC 결과
img = cv2.imread('butterfly.jpg')
ncut_avg = label2rgb(nc_labels, img_rgb, kind='avg') # N-Cut 결과
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. SLIC 슈퍼픽셀 분할 (Oversegmentation 상태, 400조각)
labels = slic(img_rgb, n_segments=400, compactness=10, sigma=1, start_label=1)

## Page 19

최적화 분할 실습
# 최종 출력 plt.subplot(1, 4, 4)
plt.figure(figsize=(20, 5)) plt.title("4. N-Cut Result")
plt.imshow(ncut_avg.astype('uint8'))
plt.subplot(1, 4, 1) plt.axis('off')
plt.title("1. Original")
plt.imshow(img_rgb) plt.tight_layout()
plt.axis('off') plt.show()

plt.subplot(1, 4, 2)
plt.title("2. SLIC Boundaries")
plt.imshow(mark_boundaries(img_rgb, labels))
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("3. SLIC Avg Color")
plt.imshow(slic_avg.astype('uint8'))
plt.axis('off')

## Page 20

영상의 품질 측정 지표

- PSNR(Peak Signal-to-Noise Ratio, 최대 신호 대 잡음비)
- 신호(Signal, 원본영상) 대비 잡음(Noise)이 얼마나 섞였는가를 수치로 나타낸 것

dB(데시벨)

- MAXI : 해당 영상의 픽셀이 가질 수 있는 최대값
(일반적인 8비트 영상에서는 255)
- MSE : 원본과 결과 영상의 픽셀 값 차이를 제곱하여 평균 낸 값

- 값이 클수록 좋은 화질(원본에 더 가까운 화질)
- 약 30dB 이상이면 보통 좋은 화질로 평가
- 인간의 시각 시스템 특성을 완전히 반영하지 못한다는 단점이 있음

## Page 21

영상의 품질 측정 지표

- PSNR(Peak Signal-to-Noise Ratio, 최대 신호 대 잡음비)
- 신호(Signal, 원본영상) 대비 잡음(Noise)이 얼마나 섞였는가를 수치로 나타낸 것

dB(데시벨)

- MAXI : 해당 영상의 픽셀이 가질 수 있는 최대값
(일반적인 8비트 영상에서는 255)
- MSE : 원본과 결과 영상의 픽셀 값 차이를 제곱하여 평균 낸 값

JPEG : 가장 많이 쓰이는 (정지)영상의 손실압축기술

## Page 22

영상의 품질 측정 지표

- SSIM(Structural Similarity Index Map, 구조적 유사도)
- 수치적인 차이보다 사람의 눈에 얼마나 비슷하게 보이는가에 초점을 맞춘 영상 품질 측정 지표
- PSNR의 한계 : PSNR은 단순히 픽셀의 값이 얼마나 달라졌는지만 측정. 그러나 우리 눈은 물체의 경계
가 뭉개지거나 노이즈가 끼는 것에 훨씬 민감
- SSIM은 원본 영상(x)과 왜곡된 영상(y)을 비교할 때 다음 세 가지 지표를 결합하여 계산

보통 α = β = γ = 1

- 휘도 (l, Luminance): 평균 밝기가 얼마나 변했는가?
- 대비 (c, Contrast): 밝기값의 표준편차가 얼마나 변했는가?
- 구조 (s, Structure): 두 영상의 픽셀 간 상관관계가 유지되는가? (물체의 형태나 패턴이 깨졌는가?)

- SSIM은 밝기가 변해도 물체의 형태(구조)가 유지되면 높은 점수를 줌. 즉, 사람의 주관적 화질 평가와
더 일치
- 약 0.9 이상이면 우수한 화질로 평가

## Page 23

영상의 품질 측정 지표

- SSIM(Structural Similarity Index Map, 구조적 유사도)
- 수치적인 차이보다 사람의 눈에 얼마나 비슷하게 보이는가에 초점을 맞춘 영상 품질 측정 지표
- PSNR의 한계 : PSNR은 단순히 픽셀의 값이 얼마나 달라졌는지만 측정. 그러나 우리 눈은 물체의 경계
가 뭉개지거나 노이즈가 끼는 것에 훨씬 민감
- SSIM은 원본 영상(x)과 왜곡된 영상(y)을 비교할 때 다음 세 가지 지표를 결합하여 계산

보통 α = β = γ = 1

## Page 24

영상의 품질 측정 실습
import urllib.request # PSNR 계산 (OpenCV 내장 함수)
from skimage.metrics import structural_similarity as ssim psnr_val = cv2.PSNR(original, restored)
from google.colab.patches import cv2_imshow
# SSIM 계산 (scikit-image 사용)
# 샘플 이미지 다운로드 ssim_val = ssim(original, restored, channel_axis=2)
url = ＇
https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg＇
# 4. 결과 출력
Urllib.request.urlretrieve(url, ＇baboon.jpg＇)
print(f"--- 품질 측정 결과 ---")
Original = cv2.imread(＇baboon.jpg＇)
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
# 영상 축소 및 재확대 (보간법 적용)
print("----------------------")
height, width = original.shape[:2]

# 가로로 붙여서 확인
# 1/4 크기로 축소 (INTER_AREA)
combined = np.hstack((original, restored))
small = cv2.resize(original, (width//4, height//4), interpolation=cv2.INTER_AREA)
print("\n[왼쪽: 원본 | 오른쪽: 1/4 축소 후 복원본]")
cv2_imshow(cv2.resize(combined, (0,0), fx=0.8, fy=0.8)) # 화면 크기에 맞춰 조절
# 다시 원래 크기로 확대 (INTER_LINEAR )
# INTER_NEAREST, INTER_CUBIC로도 실습
restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

## Page 25

내용 정리

- 영상 처리 알고리즘 실습
- 에지 및 코너 탐지
- 영상의 품질 측정 지표

## Page 26

다음 주 강의 내용

- 영상의 변환(Transform) 및 필터링
- 영상의 특징 검출 및 매칭 기술(SIFT 등)

## Page 27

(텍스트 없음)
