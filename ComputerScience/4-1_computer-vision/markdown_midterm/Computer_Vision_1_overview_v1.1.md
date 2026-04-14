# Computer_Vision_1_overview_v1.1

- Source PDF: `ComputerScience/4-1_computer-vision/Computer_Vision_1_overview_v1.1.pdf`
- Total pages: 33

## Page 1

컴퓨터 비전
Computer Vision
- Overview -

동아대학교 소프트웨어대학 AI학과
2026년 1학기

임한신

## Page 2

과목 소개

- 과목
- 컴퓨터 비전 (2026년 1학기)

- 강의 목표
- 2D 영상 처리 기술에 대한 이해
- 특징 추출 알고리즘 및 매칭 알고리즘에 대한 이해
- 기초 3D 기하 및 깊이 추정 개념 습득
- 딥러닝 기반 객체 검출 기술에 대한 이해
- 기본 영상 이해 기술에 대한 이해
- 최신 컴퓨터 비전 기술(ViT 등) 소개

- 주교재
- 컴퓨터비전과 딥러닝, 한빛출판사, 오일석 저

## Page 3

평가 방법

- 중간시험 : 40%
- 기말시험 : 40%
- 과제 : 10%
- 출석 : 10%
- 합계 : 100%

- 출석: 1회 결석 시 1점 감점, 2회 지각 시 1회 결석 처리
- 중간시험 또는 기말시험에 결석하는 경우 F 부여
- 본 강의의 수업 계획 및 내용은 사정에 따라 변경될 수 있음
- 본 교과목은 OSS 교육의 일환으로 오픈 소스 소프트웨어 도구를 활용하여 실습과 이론을 병
행합니다.
- 예) OpenCV, Pytorch 등

## Page 4

보강 안내

- 5월 5일 화요일(어린이날)
- 컴퓨터 비전 1반 보강일 : 6월 9일 화요일

- 5월 25일 월요일(석가탄신일 대체휴일)
- 컴퓨터 비전 2반 보강일 : 6월 11일 목요일

## Page 5

컴퓨터 비전이란?

- 컴퓨터 비전은 인간의 시각 능력을 컴퓨터가 모방하여 데이터를 이해하고 처리하는 기술
- 컴퓨터 비전의 입력 : 일반적으로 카메라 등의 영상 센서로부터 획득한 영상 또는 비디오
- 즉, 컴퓨터에 눈을 달아주고 눈으로 보는 것이 무엇인지를 알 수 있게 해주는 기술

- 컴퓨터 비전과 딥러닝의 관계
- 컴퓨터 비전 ≠ 딥러닝
- 딥러닝은 컴퓨터 비전의 다양한 문제를 해결하기 위한 가장 강력한 방법

## Page 6

컴퓨터 비전이란?

- 컴퓨터 비전 기술의 응용 사례

드론 의료 영상 자율주행

스마트 공장 스포츠 소비자 분석

## Page 7

컴퓨터 비전이란?

- 컴퓨터 비전 기술의 응용 사례

얼굴인식 보안 모니터링 게임 청소 로봇

지형 분석 우주 탐사 감시 휴머노이드 로봇

## Page 8

2D 영상 처리 및 분석 1D signal

- 1D signal
- 하나의 독립변수(예) 시간 t)에 따라 변하는 신호
- 음성 신호, 1D 센서 등

- 2D signal
- 두 개의 독립변수(예) 공간 좌표 x, y)에 따라 변하는 신호
- 다양한 영상 신호 등

- 영상 신호의 특징 2D signal(영상)
- 공간적 상관성이 큼. 즉, 가까이 있는 픽셀끼리는 비슷한 값을 가질 확률이 높음
- 에지(edge), 코너(corner), 텍스처(texture), 패턴(pattern)과 같은 구조적 특징이 있음

## Page 9

2D 영상 처리 및 분석

- 에지 검출(Edge detection)

Canny
edge detector

- 영역 분할(Region segmentation)

SLIC superpixels

## Page 10

2D 영상 처리 및 분석

- 기하 변환

- 샘플링(Sampling)과 보간(Interpolation)

Interpolation

## Page 11

특징 추출 및 매칭

- 특징(Feature)이란?
- 영상 내에서 특정 지점이나 영역을 다른 곳과 명확히 구분할 수 있게 해주는 유의미한 정보

- 특징의 예시
- 에지(edge)
- 밝기값이 급격히 변하는 경계선
- 코너(corner)
- 두 개 이상의 에지가 교차하는 지점. 매칭에 가장 중요
- 텍스처(texture)
- 반복되는 미세 구조
- Descriptor
- 특정 지점 주변의 픽셀 정보를 벡터 등으로 요약한 것
- Feature map
- 영상의 특징들이 어느 위치에 얼마나 강하나 나타나는지를 나타냄

## Page 12

특징 추출 및 매칭

- 좋은 특징(feature)의 조건
- 불변성(Invariance)
- 크기, 회전, 각도, 조명등의 변화에도 특징의 값이 크게 변하지 않아야 함
- 변별성(Distinctiveness)
- 코너나 패턴이 많은 영역처럼 다른 영역과 명확히 달라야 함
- 재현력(Repeatability)
- 다른 조건(다른 영상)에서도 특징으로 검출될 수 있어야 함
- ….

## Page 13

특징 추출 및 매칭

- 특징 추출(Feature extraction)
- 원본 데이터에서 의미 있는 정보를 수치 벡터 형태로 변환하는 과정
1. 검출(detection)
- 코너와 에지와 같은 특징이 될 만한 지점을 찾음
2. 특징 기술(description)
- 검출된 지점 주변의 픽셀 정보를 분석하여 descriptor라는 벡터로 변환

- 매칭(Matching)
- 둘 이상의 영상에서 같은 특징점을 찾는 과정
- 두 특징들의 descriptor 사이의 거리를 계산하여 가장 가까운 특징점을 연결
- RANSAC 등의 알고리즘을 통해 outlier를 제거하고 일관된 매칭(inlier)만 남김

RANSAC

## Page 14

3D 기하 및 깊이 추정

- 3D 좌표계(3D coordinates)

y
X1
X1 = R1X + t1
X2 X = (x, y, z, 1) X2 = R2X1 + t2
X4 X3 = R3X2 + t3
X4 = R4X3 + t4
X3
x X4 = R4(R3(R2(R1X + t1) + t2) + t3) + t4

월드 좌표계(World coordinates)
→ 3D 공간의 좌표계
z → 오른손 좌표계

## Page 15

3D 기하 및 깊이 추정

- 핀홀 카메라 모델(Pinhole camera model)

## Page 16

3D 기하 및 깊이 추정

- 깊이 추정(Depth estimation)
- 스테레오 카메라
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영
상 간의 위치 차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Left image

## Page 17

3D 기하 및 깊이 추정

- 깊이 추정(Depth estimation)
- 스테레오 카메라
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영
상 간의 위치 차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Right image

## Page 18

3D 기하 및 깊이 추정

- 깊이 추정(Depth estimation)
- 스테레오 카메라
- Disparity(양안 시차) : 두 눈(또는 두 카메라) 사이의 시점 차이로 인해 발생하는 영
상 간의 위치 차이
- 참고 : 인간의 눈은 약 6.5cm 떨어져 있음

Left image Right image

## Page 19

3D 기하 및 깊이 추정

- 3D 기하 추정(Depth estimation)
- 카메라 칼리브레이션(Camera calibration)
- 실제 카메라들의 파라미터들을 추정하는 작업
- 즉, 3D 공간에서 실제 카메라가 어디에 위치하고 어떤 방향을 향하고 있고 어디에 상
을 맺히는지 모델링하는 작업

## Page 20

3D 기하 및 깊이 추정

- 3D 기하 추정(Depth estimation)
- 카메라 칼리브레이션(Camera calibration)
- 실제 카메라들의 파라미터들을 추정하는 작업
- 즉, 3D 공간에서 실제 카메라가 어디에 위치하고 어떤 방향을 향하고 있고 어디에 상
을 맺히는지 모델링하는 작업

Camera 1
Camera 3
카메라 칼리브레이션 :
Camera 1, Camera 2, Camera 3, Camera 4
C3
C1 의 파라미터를 구함
Camera 2 Camera 4

C4
C2

## Page 21

3D 기하 및 깊이 추정

- 3D 기하 추정(Depth estimation)
- 카메라 칼리브레이션(Camera calibration)
- 실제 카메라들의 파라미터들을 추정하는 작업
- 즉, 3D 공간에서 실제 카메라가 어디에 위치하고 어떤 방향을 향하고 있고 어디에 상
을 맺히는지 모델링하는 작업
y

Camera 1
Camera 3
카메라 칼리브레이션 :
Camera 1, Camera 2, Camera 3, Camera 4
C3
C1 의 파라미터를 구함
Camera 2 Camera 4
x
C4
C2

z 월드 좌표계(World coordinates)

## Page 22

3D 기하 및 깊이 추정

- 3D 기하 추정(Depth estimation)
- COLMAP
- 여러 장의 영상으로부터 카메라 위치와 3D 장면을 복원하는 OSS
- 크게 SfM(Structure-from-Motion)과 MVS(Multi-View Stereo)의 두 단계로 이루어짐
- SfM (Structure from Motion) : 영상들 속의 특징점을 찾아 서로 매칭하고, 카메라의 파라
미터를 구함
- MVS (Multi-View Stereo) : 영상들의 모든 픽셀에 대한 깊이 정보를 계산하여 점구름
(Point Cloud)이나 메쉬(Mesh)를 생성

SfM (Structure from Motion) MVS (Multi-View Stereo)

## Page 23

객체 검출

- 객체 검출(Object detection)
- 영상에서 특정 객체의 위치와 종류를 동시에 찾는 기술

## Page 24

객체 검출

- 객체 검출(Object detection) 기술의 분류
- 2-Stage detector
- 물체가 있을 법한 후보 영역을 먼저 뽑고(1st stage) 선별
된 영역을 정밀하게 검사(2nd stage)
- 대표 기술 : Faster-RCNN 등

- 1-stage detector
- 영상 전체를 한 번만 훑고 바로 위치와 종류를 예측
- 대표 기술 : YOLO, SSD 등

- Transformer 계열 detector
- 영상 전체를 Transformer로 분석하고 object query를 이용
해 객체의 위치와 종류를 동시에 예측 객체 검출의 예
- 대표 기술 : DETR 등

## Page 25

영상 이해 기술

- 영상 분할(Image segmentation)
- 영상의 각 픽셀이 어떤 객체에 속하는지 분류하는 기술
- 의미론적 분할(Semantic segmentation)
- 영상 내의 모든 픽셀을 카테고리(클래스)별로 분류(예) 사람, 자동차, 하늘, 도로…..)
- 인스턴스 분할(Instance segmentation)
- 같은 클래스라도 개별 객체(instance)를 구분하여 분할
- 범용적 분할(Panoptic segmentation)
- 의미론적 분할과 인스턴스 분할을 합친 방식

원본영상 의미론적 분할 인스턴스 분할 범용적 분할

## Page 26

영상 이해 기술

- 자세 추정(Pose estimation)
- 영상에서 사람이나 물체의 주요 관절(키포인트, 랜드마크 )
의 위치를 추정하는 기술
- 일인 자세 추정(Single-person pose estimation)
- 한 사람의 자세만 집중적으로 추정
- 다인 자세 추정(Multi-person pose estimation)
- 여러 사람의 자세를 동시에 추정
- Top-down 방식
- 먼저 사람을 찾고 자세를 추정
- Bottom-up 방식
- 랜드마크를 모두 검출한 다음 랜드마크를 결합
하여 사람별로 자세 추정

일인 자세 추정의 예시

## Page 27

최신 컴퓨터 비전 기술

- 최신 3D 복원 기술
- 3D Gaussian Splatting
- 여러 장의 사진을 바탕으로 현실 세계의 공간이나 물체
의 가상 시점 영상을 고속/고품질로 생성하는 기술
- 3D 공간 상의 Gaussian들을 화면에 뿌려서(splatting) 영상
을 구성

3DGS의 기본 개념
3DGS 데모

## Page 28

최신 컴퓨터 비전 기술

- 최신 3D 복원 기술
- 3D Gaussian Splatting
- 3DGS의 응용
- 문화 유산, 공간의 자유시점 이동
- https://superspl.at/
- 3DGS의 동영상 적용(4DGS)
- ……. Super Splat 사이트

4DGS 데모

## Page 29

최신 컴퓨터 비전 기술

- Vision Transformer(ViT)
- 자연어 처리에 사용되었던 Transformer 구조를 영상 처리에 적용한 딥러닝 모델
- 최근 다양한 컴퓨터 비전 문제의 딥러닝 모델로 사용됨
- 적용 분야
- 영상 분류
- 객체 검출
- 영상 분할
- 영상 이해
- 깊이 추정
- 생성형 AI
- …

Vision Transformer의 기본 구조

## Page 30

최신 컴퓨터 비전 기술

- 로봇 비전(Robot vision)
- SLAM(Simultaneous Localization and Mapping)
- 로봇이나 자율 시스템이 동시에 자신의 위치를 추정하고 환경 지도를 만드는 기술
- 자율 주행의 핵심 기술로 로봇청소기, 드론, 서빙로봇 등 다양한 분야에 적용

## Page 31

내용 정리

- 컴퓨터 비전 과목 소개
- 컴퓨터 비전 기술의 적용 분야
- 컴퓨터 비전 주요 분야 개요

## Page 32

참고자료

- 참고사이트
- https://szeliski.org/Book/
- https://d2l.ai/

- 참고자료
- R. Szeliski, Computer Vision: Algorithms and Applications, 2nd ed., Springer, 2022.
- A, Torralba, P. Isola and W. T. Freeman, Foundations of Computer Vision, MIT Press, 2024

## Page 33

다음 주 강의 내용

- 2D 영상의 특성 및 영상 처리 기술
- 영상의 품질 측정 지표
- OpenCV 소개 및 실습

## Page 34

(텍스트 없음)
