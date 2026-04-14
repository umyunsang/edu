# Computer_Vision_5_Stereo_Vision_v1.0

- Source PDF: `ComputerScience/4-1_computer-vision/Computer_Vision_5_Stereo_Vision_v1.0.pdf`
- Total pages: 21

## Page 1

컴퓨터 비전
Computer Vision
- Stereo Vision -

동아대학교 소프트웨어대학 AI학과
2026년 1학기

임한신

## Page 2

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

영상평면
(Image plane)

물체에서 나온 빛은 pinhole을 일직선으로 통과하여
반대편 벽(영상평면)에 상(image)을 맺힘

## Page 3

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

카메라 중심
(Camera center)

영상평면
(Image plane)
초점거리  상하좌우가 바뀜
(Focal length)
물체에서 나온 빛은 pinhole을 일직선으로 통과하여
반대편 벽(영상평면)에 상(image)을 맺힘

## Page 4

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

카메라 중심
(Camera center)

영상평면
가상영상평면 (Image plane)
(Virtual image plane) 초점거리  상하좌우가 바뀜
 상하좌우가 동일 (Focal length)
물체에서 나온 빛은 pinhole을 일직선으로 통과하여
반대편 벽(영상평면)에 상(image)을 맺힘

## Page 5

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

카메라 중심
(Camera center)

영상평면
가상영상평면 (Image plane)
(Virtual image plane) 초점거리  상하좌우가 바뀜
 상하좌우가 동일 (Focal length)
물체에서 나온 빛은 pinhole을 일직선으로 통과하여
반대편 벽(영상평면)에 상(image)을 맺힘
가상영상평면을 두면 상하좌우가 바뀌지 않으면서
영상평면과 동일한 기하학적 모델이 가능

## Page 6

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

X = (X, Y, Z)
X = (X, Y, Z)

p=(px, py)
주축
(Principal axis) f
카메라 중심 영상평면
(Camera center) (Image plane)

x = (x, y) 는 아래와 같이 계산됨
- 주축(Principal axis) : 카메라 중심에서 영상 평면에 수직으로
𝑓X
내린 선(z-축으로 설정됨) 𝑥= + 𝑝𝑥
- 주점(Principal point) : 주축이 영상 평면과 만나는 교점. 영상 Z
의 정중앙 또는 근처에 위치 𝑓Y
𝑦= + 𝑝𝑦
Z

## Page 7

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)
X = (2X, 2Y, 2Z)

X = (X, Y, Z)

p=(px, py)
주축
(Principal axis) f
카메라 중심 영상평면
(Camera center) (Image plane)

x = (x, y) 는 아래와 같이 계산됨
- 주축(Principal axis) : 카메라 중심에서 영상 평면에 수직으로
𝑓2X 𝑓X
내린 선(z-축으로 설정됨) 𝑥= + 𝑝𝑥 = + 𝑝𝑥
- 주점(Principal point) : 주축이 영상 평면과 만나는 교점. 영상 2Z Z
의 정중앙 또는 근처에 위치 𝑓2Y 𝑓Y
𝑦= + 𝑝𝑦 = + 𝑝𝑦
2Z Z

## Page 8

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)
X = (αX, αY, αZ)

X = (X, Y, Z) 깊이 모호성
(Depth ambiguity)

p=(px, py)
주축
(Principal axis) f
카메라 중심 영상평면
(Camera center) (Image plane)

x = (x, y) 는 아래와 같이 계산됨
- 주축(Principal axis) : 카메라 중심에서 영상 평면에 수직으로
𝑓αX 𝑓X
내린 선(z-축으로 설정됨) 𝑥= + 𝑝𝑥 = + 𝑝𝑥
- 주점(Principal point) : 주축이 영상 평면과 만나는 교점. 영상 αZ Z
의 정중앙 또는 근처에 위치 𝑓αY 𝑓Y
𝑦= + 𝑝𝑦 = + 𝑝𝑦
αZ Z

## Page 9

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

깊이 모호성(Depth ambiguity)

## Page 10

카메라 모델

- 핀홀 카메라 모델(Pinhole camera model)

X = (X, Y, Z)

p=(px, py)
주축
(Principal axis) f
카메라 중심 영상평면
(Camera center) (Image plane)

카메라 내부파라미터
- 주축(Principal axis) : 카메라 중심에서 영상 평면에 수직으로 - 초점거리(Focal length(단위 : pixels))
내린 선(z-축으로 설정됨)
- 주점(Principal point(단위 : pixels))
- 주점(Principal point) : 주축이 영상 평면과 만나는 교점. 영상
의 정중앙 또는 근처에 위치

## Page 11

스테레오 비전(Stereo Vision)

- 두 개의 카메라(스테레오 카메라)를 이용해 마치 사람의 두 눈처럼 사물의 입체감과 거리(깊이,
Depth)를 추정하는 기술
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영상 간의 위치
차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Left image
스테레오 카메라

## Page 12

스테레오 비전(Stereo Vision)

- 두 개의 카메라(스테레오 카메라)를 이용해 마치 사람의 두 눈처럼 사물의 입체감과 거리(깊이,
Depth)를 추정하는 기술
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영상 간의 위치
차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Right image
스테레오 카메라

## Page 13

스테레오 비전(Stereo Vision)

- 두 개의 카메라(스테레오 카메라)를 이용해 마치 사람의 두 눈처럼 사물의 입체감과 거리(깊이,
Depth)를 추정하는 기술
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영상 간의 위치
차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Left image Right image

스테레오 카메라

## Page 14

카메라 모델

- Disparity(양안시차)와 depth

가정
- Baseline B ≠ 0
- fleft = fright = f
- 두 영상의 영상 평면은 동일한 평면 상에 있음

Depth z

영상 평면
(Image plane)
f f

Cleft Baseline B Cleft
: 두 카메라 중심 사이의 거리

## Page 15

카메라 모델

- Disparity(양안시차)와 depth

가정
- Baseline B ≠ 0
- fleft = fright = f
- 두 영상의 영상 평면은 동일한 평면 상에 있음

Depth z
Disparity d

영상 평면
(Image plane)
f f

Cleft Baseline B Cleft
: 두 카메라 중심 사이의 거리

## Page 16

카메라 모델

- Disparity(양안시차)와 depth

가정
- Baseline B ≠ 0
- fleft = fright = f
- 두 영상의 영상 평면은 동일한 평면 상에 있음

Depth z B:d=z:f
Disparity d

영상 평면 z=
(Image plane)
f f
z∝ z ∝ 𝑓B
Cleft Baseline B Cleft
: 두 카메라 중심 사이의 거리

## Page 17

카메라 모델

- Disparity(양안시차)와 depth

가정
- Baseline B ≠ 0
- fleft = fright = f
- 두 영상의 영상 평면은 동일한 평면 상에 있음

Depth z B:d=z:f
Disparity d

영상 평면 z=
(Image plane)
f f 예) f = 1000, baseline B = 65mm, d = 50

 z = 1300 mm
Cleft Baseline B Cleft
: 두 카메라 중심 사이의 거리

## Page 18

카메라 모델

- 스테레오 영상 : 일반적으로 일치점들은 영상에서 동일한 높이에 있다고 가정

Left image Right image

From https://vision.middlebury.edu/stereo/data/

## Page 19

카메라 모델

- 스테레오 영상 : 일반적으로 일치점들은 영상에서 동일한 높이에 있다고 가정

Left image Right image

From https://vision.middlebury.edu/stereo/data/

## Page 20

카메라 모델

- 스테레오 영상 : 일반적으로 일치점들은 영상에서 동일한 높이에 있다고 가정
- Disparity map : 스테레오 영상의 disparity를 픽셀 단위로 나타낸 map

Left image Right image

Disparity map of Disparity map of
left image right image

From https://vision.middlebury.edu/stereo/data/

## Page 21

Disparity Estimation(양안시차 추정)

- Disparity estimation(양안시차 추정)  depth estimation(깊이 추정)
- SIFT, ORB 등 특징 추출 기술을 이용하는 경우
 특징이 아니 대부분의 픽셀의 disparity 추정이 어려움

## Page 22

(텍스트 없음)
