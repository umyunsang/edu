## --- [Page 1] ---
1/33

Dong-A Univ. (ISPL)

컴퓨터AI공학부 AI학과

2025년 1학기 머신러닝


| [실습] Super Resolution | using CNN |  |
| --- | --- | --- |


## --- [Page 2] ---
2/33

Super Resolution (SR)

▪Single Image SR (SISR)

•
저해상도 이미지 1개를 입력 받아 고해상도 이미지 1개 출력

▪Multi Image SR (Video SR)

•
저해상도 이미지 여러 개를 입력 받아 고해상도 이미지 1개또는 여러 개 출력

## --- [Page 3] ---
3/33

Super Resolution (SR) - 일반적인 SR 기법

▪Interpolation-based SR

•
픽셀 사이의 값을 예측해 고해상도 이미지로 출력

Ref.: https://en.wikipedia.org/wiki/Bicubic_interpolation

: 실제 픽셀 값

: 예측 픽셀 값

픽셀 위치

픽셀 값

## --- [Page 4] ---
4/33

Super Resolution (SR) - 일반적인 SR 기법

▪Interpolation-based SR

•
픽셀 사이의 값을 예측해 고해상도 이미지로 출력

1D signal interpolation
(Ex. Audio)

2D signal interpolation
(Ex. Image)

Ref.: https://en.wikipedia.org/wiki/Bicubic_interpolation

## --- [Page 5] ---
5/33

Super Resolution (SR) - 일반적인 SR 기법

▪Interpolation-based SR

•
픽셀 사이의 값을 예측해 고해상도 이미지로 출력

Ref.: https://en.wikipedia.org/wiki/Bicubic_interpolation

## --- [Page 6] ---
6/33

Super Resolution (SR) - 일반적인 SR 기법

▪Interpolation-based SR

•
참조 범위 내의 픽셀들의 평균 값을 참조하기 때문에 부드럽고 자연스러운 영상을 생성

•
평균값 연산의 smoothing 효과로 인해 영상의 선명도가 떨어지는 단점이 있음

•
특히 고주파 신호의 복원력이 떨어짐

Ref.: https://en.wikipedia.org/wiki/Bicubic_interpolation

Bi-cubic
interpolation

## --- [Page 7] ---
7/33

Super Resolution (SR) - 일반적인 SR 기법

▪Example-based SR: 현재 이미지 또는 다른 이미지의 패치를 이용해 SR 적용

Ref.: [CVPR 2015] SISR from Transformed Self-Exemplars

자기유사성 기반 SR
External Database 기반 SR

## --- [Page 8] ---
8/33

Super Resolution (SR) - 일반적인 SR 기법

▪Example-based SR: 현재 이미지 또는 다른 이미지의 패치를 이용해 SR 적용

•
우수한 성능을 위해서는 방대한 양의 데이터베이스가 필요함 →검색 시간이 오래 걸림

•
데이터베이스 내에 적절한 match를 찾지 못할 경우 주관적 화질이 크게 떨어질 수 있음

Ref.: [CVPR 2015] SISR from Transformed Self-Exemplars

자기유사성 기반 SR
External Database 기반 SR

## --- [Page 9] ---
9/33

Super Resolution (SR) - 일반적인 SR 기법

▪SR 기술 연구 동향

Interpolation based

Example based

Self-similarity
Machine Learning

Deep Learning
Misc.

CNN
GAN
최신 기술

## --- [Page 10] ---
10/33

Super Resolution (SR) - SR 응용 분야

▪SR 기술 응용 분야

2007년 방영
(1280 x 720)

2018년 재방영
(3840 x 2160)

## --- [Page 11] ---
11/33

Super Resolution (SR) - SR 응용 분야

▪Artifact Reduction (AR): 압축으로 발생하는 잡음을 제거하는 기술

JPEG 압축 이미지
ARCNN 복원 이미지

Ref.: [IVMSPW] 3D Image Quality Index using SDP based Binocular Perception Model

[ICCV 2015] Compression Artifact Reduction by a DCN

## --- [Page 12] ---
12/33

Super Resolution (SR) - SR 응용 분야

▪Image colorization: 흑백 이미지를 입력 받아 채색된 이미지 출력

Ref.: [arXiv 2018] ColorUNet: A convolutional classification approach to colorization

## --- [Page 13] ---
13/33

Super Resolution (SR) - SR 응용 분야

▪Image style transfer

Content image

Style image

Ref.: [CVPR 2016] Image Style Transfer Using CNN

## --- [Page 14] ---
14/33

Super Resolution (SR) - SR 응용 분야

▪Image style transfer: https://reiinakano.com/arbitrary-image-stylization-tfjs/

## --- [Page 15] ---
15/33

SR 성능 평가 방법

▪객관적 성능 평가

•
최대 신호 대 잡음 비(Peak Signal-to-Noise Ratio, PSNR)

•
구조적 유사 지수(Structural Similarity Index Measure, SSIM)

•
다중 스케일 구조적 유사 지수(Multi-Scale SSIM, MS-SSIM)

▪주관적 성능 평가

•
평균 주관 점수 (Mean Opinion Score, MOS)

## --- [Page 16] ---
16/33

SR 성능 평가 방법 - 객관적 성능 평가 방법

▪Peak Signal-to-Noise Ratio (PSNR)

•
각 픽셀 간 차이 (MSE)를 이용해 계산

•
Log scale 값이므로 [dB] 단위 사용

(
)

2

1

1
MSE

w h

i
i
i

O
R
w h



=
=
−


2

10
10
PSNR
10 log
20 log
MSE
MSE
MAX
MAX




=

=










❖MSE: Mean Square Error
❖PSNR: Peak Signal-to-Noise Ratio

8bit 영상인 경우 255

## --- [Page 17] ---
17/33

SR 성능 평가 방법 - 주관적 성능 평가 방법

▪Mean Opinion Score (MOS)

•
사람이 직접 품질에 대한 점수 부여, 평균값을 사용

SRGAN의 MOS 평가 결과

Ref.: [CVPR 2016] Photo-Realistic SISR Using a Generative Adversarial Network

MOS 평가 환경 예시

❖
IRB: 기관생명 윤리위원회

## --- [Page 18] ---
18/33

Super Resolution using Convolutional Neural Network (SRCNN)

▪[IEEE TPAMI 2015] Dong et al. (PKU)

▪Image-input, Image-output 구조의 3-layer CNN

▪2/3/4배 해상도로 출력하게 학습 (Ex. 3배 모델 →입력: 32x32, 출력: 96x96)

[9x9x3]x64
[1x1x64]x32
[5x5x32]x3

## --- [Page 19] ---
19/33

Super Resolution using Convolutional Neural Network (SRCNN)

▪SRCNN 입출력 구조

•
저해상도 이미지 (LR)를 interpolation한 이미지 (ILR)를 입력

❖LR: Low Resolution
❖ILR: Interpolated LR
❖HR: High Resolution (Original)

Original image
High Resolution (HR)

Low Resolution (LR)
Interpolated LR (ILR)

Down
sampling

Bi-cubic

## --- [Page 20] ---
20/33

Super Resolution using Convolutional Neural Network (SRCNN)

▪SRCNN 데이터셋 구성 방법

❖LR: Low Resolution
❖ILR: Interpolated LR
❖HR: High Resolution (Original)

Original image
High Resolution (HR)

Low Resolution (LR)
Interpolated LR (ILR)

Down
sampling

Bi-cubic

SRCNN 입력 데이터
SRCNN 정답 데이터

## --- [Page 21] ---
21/33

Super Resolution using Convolutional Neural Network (SRCNN)

▪SRCNN 학습 방법

LR image

입력 이미지

SRCNN

Pred. image

예측 이미지

## --- [Page 22] ---
22/33

Super Resolution using Convolutional Neural Network (SRCNN)

▪SRCNN 학습 방법

LR image

입력 이미지

SRCNN

Pred. image

예측 이미지

HR image

정답 이미지

▪Train

✓Loss function 정의

✓최적화

▪
Test

✓PSNR 측정

✓MOS 측정

## --- [Page 23] ---
23/33

SRCNN - SR dataset 구성 방법

▪Training dataset 구성(Scale: 4)

•
T-91 이미지 데이터셋 (91장)

•
하나의 이미지에서 32x32 단위 patch로 나누어 dataset 구성

▪Testing dataset 구성

•
Set5 이미지 데이터셋 (5장)

T-91 dataset
Set5 dataset

## --- [Page 24] ---
24/33

SRCNN - SR dataset 구성 방법

▪Training dataset 구성(Scale: 4)

•
T-91 이미지 데이터셋 (91장)

•
하나의 이미지에서 32x32 단위 patch로 나누어 dataset 구성

▪Testing dataset 구성

•
Set5 이미지 데이터셋 (5장)

ILR Image (Input)

32

32

Training dataset 구성 예시

HR Image (Label)

32

32

❖LR: Low Resolution
❖ILR: Interpolated LR
❖HR: High Resolution (Original)

## --- [Page 25] ---
25/33

SRCNN - SR dataset 구성 방법

▪Training dataset 구성(Scale: 4)

•
T-91 이미지 데이터셋 (91장)

•
하나의 이미지에서 32x32 단위 patch로 나누어 dataset 구성

▪Testing dataset 구성

•
Set5 이미지 데이터셋 (5장)

ILR Image (Input)

32

32

Training dataset 구성 예시

HR Image (Label)

32

32

❖LR: Low Resolution
❖ILR: Interpolated LR
❖HR: High Resolution (Original)

Data 1

## --- [Page 26] ---
26/33

SRCNN - SR dataset 구성 방법

▪Training dataset 구성(Scale: 4)

•
T-91 이미지 데이터셋 (91장)

•
하나의 이미지에서 32x32 단위 patch로 나누어 dataset 구성

▪Testing dataset 구성

•
Set5 이미지 데이터셋 (5장)

ILR Image (Input)

32

32

Training dataset 구성 예시

32

32

❖LR: Low Resolution
❖ILR: Interpolated LR
❖HR: High Resolution (Original)

Data 1
Data 2


|  |  |
| --- | --- |
| H |  |

## --- [Page 27] ---
27/33

SRCNN - SR dataset 구성 방법

▪SR data loader 정의

•
(4) Training data loader 정의

✓__init__(self): 데이터셋 전처리 (이미지 패치 수행)

✓__len__(self): 데이터셋 개수 반환

✓__getitem__(self, idx): idx 번째 데이터 반환

## --- [Page 28] ---
28/33

SRCNN - SR dataset 구성 방법

▪SR data loader 정의

•
(5) Testing data loader 정의

➢주의사항: Test dataset은 이미지 패치를 수행하지 않음

## --- [Page 29] ---
29/33

SRCNN - SR 모델 학습

▪SRCNN 모델 학습

•
(1) SRCNN 모델 정의

: Convolution layer

: Activation function

[9x9x3]x64, s1, p4

ReLU

[1x1x64]x32, s1, p2

ReLU

[5x5x32]x3, s1, p2

ILR
image

Pred.
image

## --- [Page 30] ---
30/33

SRCNN - SR 모델 학습

▪SRCNN 모델 학습

•
(3) Training loop 선언

## --- [Page 31] ---
31/33

SRCNN - SR 성능 검증

▪SRCNN 모델 학습

•
(7) 구글 드라이브 데이터셋 폴더에 저장된 복원 이미지 확인

Original (HR)
Bicubic interpolation (ILR)
SRCNN (Pred.)
30.4304dB
31.1327dB

## --- [Page 32] ---
32/33

SRCNN - SR 성능 검증

▪SRCNN 모델 학습

•
(7) 구글 드라이브 데이터셋 폴더에 저장된 복원 이미지 확인

Original (HR)
Bicubic interpolation (ILR)
SRCNN (Pred.)
30.4304dB
31.1327dB

## --- [Page 33] ---
33/33

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea