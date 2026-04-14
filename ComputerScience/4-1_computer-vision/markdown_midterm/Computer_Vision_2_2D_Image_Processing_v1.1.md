# Computer_Vision_2_2D_Image_Processing_v1.1

- Source PDF: `ComputerScience/4-1_computer-vision/Computer_Vision_2_2D_Image_Processing_v1.1.pdf`
- Total pages: 19

## Page 1

명암 조절

- 화면의 밝기(Luminance)를 밝거나 어둡게 조정
- 감마 조절(gamma correction)
- 화면의 밝기를 인간의 시각 특성에 맞게 비선형적으로 조절하는 과정
- 사람은 어두운 곳에서의 변화에는 민감하게 반응하지만, 아주 밝은 곳에서의 변화에는
상대적으로 둔감
- 아래의 수식에서 γ (감마, gamma)값으로 화면의 밝기를 조절

L = 보통 256
𝑓̇ : 0~255 사이 밝기를 0~1사이로 정규화

## Page 2

명암 조절

- 화면의 밝기(Luminance)를 밝거나 어둡게 조정
- 감마 조절(gamma correction)

γ = 0.5 γ=1 γ=2

## Page 3

명암 조절

- 감마 조절(gamma correction)
- 일반적인 모니터는 화면을 어둡게 출력하는 성질(γ ≈ 2.2)이 있음
- 이를 상쇄하기 위해 영상을 만들 때 미리 화면을 그만큼 밝게(γ ≈ 1/2.2) 만들어서 저장
- 두 과정을 거치면 ½.2 x 2.2 ≈ 1 이 되어, 우리 눈에 원래의 자연스러운 밝기로 보이게 됨

## Page 4

모폴로지(Morphology)

- 이미지 내에서 객체의 모양이나 구조를 분석하고 조작하는 기법
- 구조 요소(structuring element)를 이용하여 영역의 모양을 조작

구조요소의 예
- 기본 연산
- 팽창(dilation) : 작은 홈을 메우거나 끊어진 영역을 연결하는 효과. 영역을 키움
- 침식(erosion) : 경계에 솟은 돌출 부분을 깎는 효과. 영역을 작게 만듦
- 열림(opening) : 침식한 결과에 팽창 적용. 원래 영역 크기 유지
- 닫힘(closing) : 팽창한 결과에 침식을 적용. 원래 영역 크기 유지

## Page 5

모폴로지(Morphology) 예시

구조 요소

입력 영상

팽창

잘라낸 영상 팽창 침식 닫힘

침식

## Page 6

기하 변환

- Homogeneous coordinates(동차 좌표계)
- 2차원 좌표에 1을 추가해 3차원 벡터로 표현
- 3개 요소에 같은 값을 곱하면 같은 좌표. 예) (-2,4,1)과 (-4,8,2)는 (-2,4)에 해당

(0, 0) x
영상 평면
(Image plane)

y
(100, 100)

(Height-1, Width-1)

## Page 7

기하 변환

- Homogeneous coordinates(동차 좌표계)
- 2차원 좌표에 1을 추가해 3차원 벡터로 표현
- 3개 요소에 같은 값을 곱하면 같은 좌표. 예) (-2,4,1)과 (-4,8,2)는 (-2,4)에 해당

(0, 0) x
영상 평면 (100, 100)
(Image plane)
Homogeneous coordinates(동차 좌표계)
y
(100, 100) (100, 100, 1)
(200, 200, 2)
(150, 150, 1.5) 동일한 좌표
… (k : scale factor)
(Height-1, Width-1)
(100k, 100k, k)

## Page 8

기하 변환

- 기하 변환의 종류

x 방향으로 tx, y 방향으로
이동
ty 만큼 이동

회전 원점을 중심으로 반시계 방향으로
θ만큼 회전

크기 x 방향으로 sx, y 방향으로
sy 만큼 크기 조정

## Page 9

기하 변환

- 기하 변환의 예
- 정사각형을 x 방향으로 2, y 방향으로 -1만큼 이동한 다음 반시계 방향으로 30도 회전

이동 적용

회전 적용

## Page 10

기하 변환

- 기하 변환의 예
- 정사각형을 x 방향으로 2, y 방향으로 -1만큼 이동한 다음 반시계 방향으로 30도 회전

- 복합 변환을 위한 행렬을 미리 곱해 놓으면 모든 점에
대해 한번의 행렬 곱셈으로 기하 변환 가능

## Page 11

기하 변환

- Forward mapping and Backward mapping
- Forward mapping 때는 변환 영상에서 값을 받지 못하는 픽셀이 생기는 문제 발생
- Backward mapping 방식으로 문제 해결 가능

Forward mapping (전방 변환) Backward mapping (후방 변환)

## Page 12

OpenCV 개요

- 컴퓨터 비전과 이미지 처리를 목적으로 하는 오픈소스 라이브러리
- 2000년 인텔리서치에서 공개
- 현재는 OpenCV Foundation에서 비영리 운영
- 함수와 클래스는 C/C++로 개발
- 인터페이스 언어는 C, C++, 자바, 자바스크립트, 파이썬 지원
- 교육 목적, 상용 목적 모두 무료 사용 가능(Apache 2.0)
- 가장 최신 버전은 OpenCV 4.x
- 공식 홈페이지 : https://opencv.org
- 매뉴얼 사이트 : https://docs.opencv.org/
- 공식 깃허브 : https://github.com/opencv/opencv

## Page 13

실습 환경

- 구글 코랩(Google Colab)
- 코랩에는 opencv-python 라이브러리가 이미 설치되어 있음
- cv2.imshow() 함수를 사용하면 에러가 발생하므로 코랩 전용 출력 함수를 사용해야 함
- 웹상의 이미지를내려받거나 영상 파일을 업로드해서 사용 가능

# OpenCV의 version 확인
!pip show opencv-python

## Page 14

영상처리 실습

- OpenCV 영상 출력
import urllib.request
from google.colab.patches import cv2_imshow

# URL 설정 (OpenCV 공식 깃허브 저장소 내 샘플 데이터)
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg'
save_name = ‘fruits.jpg'

# 파일 다운로드
urllib.request.urlretrieve(url, save_name)

# 확인 출력
img = cv2.imread(save_name)
cv2_imshow(img)

Bicubic interpolation

## Page 15

영상처리 실습

- OpenCV에서 영상은 numpy.ndarray 클래스 형의 객체임
- 그러므로 OpenCV가 다루는 영상은 numpy가 제공하는 다양한 기능(함수)을 사용할 수 있음
- Matplotlib으로 시각화하거나 Scikit-learn, PyTorch, TensorFlow 같은 딥러닝/머신러닝 라이
브러리에 바로 적용 가능
- NumPy의 슬라이싱 기능 등을 사용하면 이미지의 특정 영역(ROI, Region of Interest)을 잘
라내거나 수정하는 작업 용이
- Numpy는 내부적으로 C와 포트란으로 구현되어 있어 파이썬의 기본 리스트보다 매우 빠름
pip show numpy # numpy가 설치되어 있는지 확인
import numpy as np #numpy import

# 영상의 배열 정보 확인
img.shape # (480, 512, 3)
img.ndim #3
img.size # 737280
img.dtype # dtype(‘uint8')

## Page 16

영상처리 실습

- OpenCV에서의 영상 데이터 구조

※ OpenCV에서는 BGR의 순서로 픽셀 데이터를 저장

print(img[0,0,0], img[0,0,1], img[0,0,2]) # (0, 0) 픽셀값

print(img[0,1,0], img[0,1,1], img[0,1,2]) # (0, 1) 픽셀값

## Page 17

영상처리 실습

- OpenCV에서의 슬라이싱(Slicing)

B_channel = img[ :, :, 0] # blue channel roi = img[100:300, 200:400]
G_channel = img[ :, :, 1] # green channel cv2_imshow(roi)
R_channel = img[ :, :, 2] # red channel
downsampled = img[::2, ::2]
cv2_imshow(R_channel) cv2_imshow(downsampled)
cv2_imshow(G_channel)
cv2_imshow(B_channel)

## Page 18

내용 정리

- 2D 영상의 특성 및 영상 처리 기술
- 영상의 품질 측정 지표
- OpenCV 소개 및 영상 데이터 구조

## Page 19

다음 주 강의 내용

- 영상의 변환(Transform) 및 필터링
- 에지 및 코너 탐지
- 영상의 품질 측정 지표

## Page 20

(텍스트 없음)
