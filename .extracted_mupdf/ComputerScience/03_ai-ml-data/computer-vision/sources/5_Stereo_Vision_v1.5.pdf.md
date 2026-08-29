## --- [Page 1] ---
컴퓨터비전
Computer Vision

- Stereo Vision -

동아대학교소프트웨어대학AI학과

2026년1학기

임한신

## --- [Page 2] ---
• 핀홀카메라모델(Pinhole camera model)

물체에서나온빛은pinhole을일직선으로통과하여
반대편벽(영상평면)에상(image)을맺힘

영상평면
(Image plane)


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 3] ---
• 핀홀카메라모델(Pinhole camera model)

물체에서나온빛은pinhole을일직선으로통과하여
반대편벽(영상평면)에상(image)을맺힘

카메라중심
(Camera center)

초점거리
(Focal length)

영상평면
(Image plane)
상하좌우가바뀜


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 4] ---
• 핀홀카메라모델(Pinhole camera model)

물체에서나온빛은pinhole을일직선으로통과하여
반대편벽(영상평면)에상(image)을맺힘

카메라중심
(Camera center)

초점거리
(Focal length)

가상영상평면
(Virtual image plane)
상하좌우가동일

영상평면
(Image plane)
상하좌우가바뀜


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 5] ---
• 핀홀카메라모델(Pinhole camera model)

물체에서나온빛은pinhole을일직선으로통과하여
반대편벽(영상평면)에상(image)을맺힘
가상영상평면을두면상하좌우가바뀌지않으면서
영상평면과동일한기하학적모델이가능

카메라중심
(Camera center)

초점거리
(Focal length)

가상영상평면
(Virtual image plane)
상하좌우가동일

영상평면
(Image plane)
상하좌우가바뀜


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 6] ---
• 핀홀카메라모델(Pinhole camera model)

X = (X, Y, Z)

p=(px, py)
p=(px, py)

X = (X, Y, Z)

𝑥= 𝑓X

Z + 𝑝𝑥

𝑦= 𝑓Y

Z + 𝑝𝑦

x = (x, y) 는아래와같이계산됨

f
카메라중심
(Camera center)

영상평면
(Image plane)

주축
(Principal axis)

•
주축(Principal axis) : 카메라중심에서영상평면에수직으로
내린선(z-축으로설정됨)
•
주점(Principal point) : 주축이영상평면과만나는교점. 영상
의정중앙또는근처에위치


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 7] ---
• 핀홀카메라모델(Pinhole camera model)
X = (2X, 2Y, 2Z)

p=(px, py)
p=(px, py)

𝑥= 𝑓2X

2Z + 𝑝𝑥= 𝑓X
Z + 𝑝𝑥

𝑦= 𝑓2Y

2Z + 𝑝𝑦= 𝑓Y
Z + 𝑝𝑦

X = (X, Y, Z)

x = (x, y) 는아래와같이계산됨

f
카메라중심
(Camera center)

영상평면
(Image plane)

주축
(Principal axis)

•
주축(Principal axis) : 카메라중심에서영상평면에수직으로
내린선(z-축으로설정됨)
•
주점(Principal point) : 주축이영상평면과만나는교점. 영상
의정중앙또는근처에위치


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 8] ---
• 핀홀카메라모델(Pinhole camera model)
X = (αX, αY, αZ)

p=(px, py)
p=(px, py)

𝑥= 𝑓αX

αZ + 𝑝𝑥= 𝑓X

Z + 𝑝𝑥

𝑦= 𝑓αY

αZ + 𝑝𝑦= 𝑓Y

Z + 𝑝𝑦

X = (X, Y, Z)

x = (x, y) 는아래와같이계산됨

f

깊이모호성
(Depth ambiguity)

카메라중심
(Camera center)

영상평면
(Image plane)

주축
(Principal axis)

•
주축(Principal axis) : 카메라중심에서영상평면에수직으로
내린선(z-축으로설정됨)
•
주점(Principal point) : 주축이영상평면과만나는교점. 영상
의정중앙또는근처에위치


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 9] ---
• 핀홀카메라모델(Pinhole camera model)

깊이모호성(Depth ambiguity)


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 10] ---
• 핀홀카메라모델(Pinhole camera model)

f

p=(px, py)

X = (X, Y, Z)

카메라중심
(Camera center)

영상평면
(Image plane)

주축
(Principal axis)

•
주축(Principal axis) : 카메라중심에서영상평면에수직으로
내린선(z-축으로설정됨)
•
주점(Principal point) : 주축이영상평면과만나는교점. 영상
의정중앙또는근처에위치

카메라내부파라미터
•
초점거리(Focal length(단위: pixels))
•
주점(Principal point(단위: pixels))


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 11] ---
• 핀홀카메라모델(Pinhole camera model)

f

카메라중심
(Camera center)

영상평면
(Image plane)

주축
(Principal axis)

•
주축(Principal axis) : 카메라중심에서영상평면에수직으로
내린선(z-축으로설정됨)
•
주점(Principal point) : 주축이영상평면과만나는교점. 영상
의정중앙또는근처에위치

f

px= W/2

θ

수평화각= 2θ = 2×arctan(W / 2f)

예) W : 1920,  f : 약1330 라고하면

수평화각은약71.6도정도나옴

카메라화각계산

C

Z

영상평면
(Image plane)


|  | 카메라 모델 |  |
| --- | --- | --- |


| X = (X, Y, | Z) |
| --- | --- |


## --- [Page 12] ---
• 두개의카메라(스테레오카메라)를이용해마치사람의두눈처럼사물의입체감과거리(깊이, 
Depth)를추정하는기술

• Disparity(양안시차) : 두눈(또는두카메라) 사이의시점차이로인해발생하는영상간의위치
차이

• 참고: 인간의눈은약6.5cm 떨어져있음

Left image

스테레오카메라


|  | 스테레오 비전(Stereo Vision) |  |
| --- | --- | --- |


## --- [Page 13] ---
• 두개의카메라(스테레오카메라)를이용해마치사람의두눈처럼사물의입체감과거리(깊이, 
Depth)를추정하는기술

• Disparity(양안시차) : 두눈(또는두카메라) 사이의시점차이로인해발생하는영상간의위치
차이

• 참고: 인간의눈은약6.5cm 떨어져있음

스테레오카메라

Right image


|  | 스테레오 비전(Stereo Vision) |  |
| --- | --- | --- |


## --- [Page 14] ---
• 두개의카메라(스테레오카메라)를이용해마치사람의두눈처럼사물의입체감과거리(깊이, 
Depth)를추정하는기술

• Disparity(양안시차) : 두눈(또는두카메라) 사이의시점차이로인해발생하는영상간의위치
차이

• 참고: 인간의눈은약6.5cm 떨어져있음

스테레오카메라

Right image
Left image


|  | 스테레오 비전(Stereo Vision) |  |
| --- | --- | --- |


## --- [Page 15] ---
• Disparity(양안시차)와depth

Cleft
Cleft

f
f

Baseline B
: 두카메라중심사이의거리

Depth z

영상평면
(Image plane)

가정

•
Baseline B ≠0

•
fleft = fright = f

•
두영상의영상평면은동일한평면상에있음


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 16] ---
• Disparity(양안시차)와depth

Cleft
Cleft

f
f

Baseline B
: 두카메라중심사이의거리

Disparity d

Depth z

영상평면
(Image plane)

가정

•
Baseline B ≠0

•
fleft = fright = f

•
두영상의영상평면은동일한평면상에있음


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 17] ---
• Disparity(양안시차)와depth

Cleft
Cleft

f
f

Baseline B
: 두카메라중심사이의거리

Disparity d

Depth z

영상평면
(Image plane)

B : d = z : f

z =

௙୆

ௗ

가정

•
Baseline B ≠0

•
fleft = fright = f

•
두영상의영상평면은동일한평면상에있음

z ∝

ଵ
ௗ
z ∝𝑓B


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 18] ---
• Disparity(양안시차)와depth

Cleft
Cleft

f
f

Baseline B
: 두카메라중심사이의거리

Disparity d

Depth z

영상평면
(Image plane)

B : d = z : f

z =

௙୆

ௗ

가정

•
Baseline B ≠0

•
fleft = fright = f

•
두영상의영상평면은동일한평면상에있음

예) f = 1000, baseline B = 65mm, d = 50

z = 1300 mm


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 19] ---
• 스테레오영상: 일반적으로일치점들은영상에서동일한높이에있다고가정

From https://vision.middlebury.edu/stereo/data/

Left image
Right image


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 20] ---
• 스테레오영상: 일반적으로일치점들은영상에서동일한높이에있다고가정

From https://vision.middlebury.edu/stereo/data/

Left image
Right image


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 21] ---
• 스테레오영상: 일반적으로일치점들은영상에서동일한높이에있다고가정

• Disparity map(양안시차지도, 깊이지도, 깊이맵) : 스테레오영상의disparity를픽셀단위로나타낸map

From https://vision.middlebury.edu/stereo/data/

Left image
Right image

Disparity map of 
left image

Disparity map of 
right image


|  | 카메라 모델 |  |
| --- | --- | --- |


## --- [Page 22] ---
• Disparity estimation(양안시차추정) depth estimation(깊이추정)

• SIFT, ORB 등특징추출기술을이용하는경우

특징이아니대부분의픽셀의disparity 추정이어려움


|  | Disparity Estimation(양안시차 |  |
| --- | --- | --- |


## --- [Page 23] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• SAD, SSD, NCC 등

• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지를최소화하는방식으로깊이지도생성
• GraphCut(그래프기반방법) 등

• Semi-global matching(세미-전역매칭) 기반방법
• 블록매칭의속도와전역매칭의정확도사이에서균형을맞춘방식
• SGBM(Semi-Global Block Matching) 등

• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• RAFT-Stereo 등


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 24] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• SAD, SSD, NCC 등

(x, y)
(x−d, y)

Left image
Right image

d : disparity


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 25] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• SAD, SSD, NCC 등

(x, y)
(x−d, y)

Left image
Right image

: 블록

d : disparity


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 26] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• SAD, SSD, NCC 등
• 주요파라미터: 블록사이즈, search range

(x, y)
(x−d, y)

Left image
Right image

: 블록

d : disparity

: Search range


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 27] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• SAD(Sum of Absolute Difference)
• 대응하는픽셀값차이의절댓값을모두더함
• 연산량이가장적음

• SSD(Sum of Squared Difference)
• 픽셀값차이를제곱하여더함
• 차이가큰픽셀(노이즈)에대해더큰페널티를부여

3x3 인블록(W)의(i, j)예시


|  | Disparity Estimation |  |
| --- | --- | --- |


| -1, -1 | 0, -1 | 1, -1 |
| --- | --- | --- |
| -1, 0 | 0, 0 | 1, 0 |
| -1, 1 | 0, 1 | 1, 1 |

## --- [Page 28] ---
• Block matching(블록매칭) 기반방법
• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식
• NCC(Normalized Cross Correlation)
• 분자에는두블록사이의공분산, 분모에는각블록의표준편차의곱이
들어감
• -1 ≤NCC ≤1이며, 1에가까울수록두패치가완벽하게일치함을뜻함
• 두블록의상관계수(correlation coefficient)를구하는것과같음

두블록사이의공분산

각블록의표준편차의곱

3x3 인블록(W)의(i, j)예시


|  | Disparity Estimation |  |
| --- | --- | --- |


| -1, -1 | 0, -1 | 1, -1 |
| --- | --- | --- |
| -1, 0 | 0, 0 | 1, 0 |
| -1, 1 | 0, 1 | 1, 1 |

## --- [Page 29] ---
• Block matching(블록매칭) 기반방법

• 픽셀주변의작은윈도우(Block) 단위로유사도를비교하는방식

• 블록매칭의장점

• 알고리즘이직관적이고구현이비교적쉬움. 즉, 왼쪽영상의블록을오른쪽영상의같은행
(Scanline)을따라이동시키며유사도(SAD, SSD 등)를계산하기만하면됨

• 연산구조가병렬처리에매우유리함. GPU나FPGA에서수천개의블록을동시에계산할수있어
실시간(Real-time) 처리가가능

• 특정탐색범위내의데이터만로드하면되므로메모리점유율이비교적낮음


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 30] ---
• Block matching(블록매칭) 기반방법

• 블록매칭의단점

• 텍스처부족영역(Textureless Region) 취약: 단색벽면이나하늘처럼특징이없는곳에서는모든블
록이비슷함.

• 반복패턴(Repetitive Patterns) 취약: 체커보드나창문격자처럼같은모양이반복되는곳에서는어
느영역이진짜쌍인지구분이어려움

• 가려짐(Occlusion) 문제취약: 한쪽카메라에는보이지만다른쪽카메라에는가려져서보이지않
는영역(Occlusion region)에대해서는블록매칭으로disparity 추정이어려움

텍스처부족영역(Textureless Region)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 31] ---
• Block matching(블록매칭) 기반방법

• 블록매칭의단점

• 텍스처부족영역(Textureless Region) 취약: 단색벽면이나하늘처럼특징이없는곳에서는모든블
록이비슷함.

• 반복패턴(Repetitive Patterns) 취약: 체커보드나창문격자처럼같은모양이반복되는곳에서는어
느영역이진짜쌍인지구분이어려움

• 가려짐(Occlusion) 문제취약: 한쪽카메라에는보이지만다른쪽카메라에는가려져서보이지않
는영역(Occlusion region)에대해서는블록매칭으로disparity 추정이어려움

반복패턴(Repetitive Patterns)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 32] ---
• Block matching(블록매칭) 기반방법

• 블록매칭의단점

• 텍스처부족영역(Textureless Region) 취약: 단색벽면이나하늘처럼특징이없는곳에서는모든블
록이비슷함.

• 반복패턴(Repetitive Patterns) 취약: 체커보드나창문격자처럼같은모양이반복되는곳에서는어
느영역이진짜쌍인지구분이어려움

• 가려짐(Occlusion) 문제취약: 한쪽카메라에는보이지만다른쪽카메라에는가려져서보이지않
는영역(Occlusion region)에대해서는블록매칭으로disparity 추정이어려움

가려짐영역(Occlusion region)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 33] ---
• Block matching(블록매칭) 기반방법

• 블록매칭의단점

• 텍스처부족영역(Textureless Region) 취약: 단색벽면이나하늘처럼특징이없는곳에서는모든블
록이비슷함.

• 반복패턴(Repetitive Patterns) 취약: 체커보드나창문격자처럼같은모양이반복되는곳에서는어
느영역이진짜쌍인지구분이어려움

• 가려짐(Occlusion) 문제취약: 한쪽카메라에는보이지만다른쪽카메라에는가려져서보이지않
는영역(Occlusion region)에대해서는블록매칭으로disparity 추정이어려움

가려짐영역(Occlusion region)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 34] ---


|  | Blocking Matching 실습 |  |
| --- | --- | --- |


| import cv2 import numpy as np import time import matplotlib.pyplot as plt import urllib.request # Aloe 이미지다운로드 url1 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg' url2 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg' urllib.request.urlretrieve(url1, 'aloeL.jpg') urllib.request.urlretrieve(url2, 'aloeR.jpg') imgL color = cv2.imread('aloeL.jpg') _ imgL gray = cv2.imread('aloeL.jpg', cv2.IMREAD GRAYSCALE) _ _ imgR gray = cv2.imread('aloeR.jpg', cv2.IMREAD GRAYSCALE) _ _ # 원본해상도저장 orig h, orig w = imgL gray.shape _ _ _ # StereoBM 설정(SAD 기반) num disp = 128 _ block size = 15 _ | stereo = cv2.StereoBM create(numDisparities=num disp, blockSize=block size) _ _ _ # disparity 계산 start time = time.time() _ disparity = stereo.compute(imgL gray, imgR gray) _ _ elapsed time = time.time() - start time _ _ # StereoBM 결과는16배스케일이므로실제픽셀단위로변환 disparity float = disparity.astype(np.float32) / 16.0 _ # 시각화를위한정규화(0~255) # 시차값이0보다작은영역(무효값)은제외 disparity vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM MINMAX, _ _ cv2.CV 8U) _ # 결과출력 print(f"이미지크기: {orig w}x{orig h}") _ _ print(f"소요시간: {elapsed time:.4f}초") _ plt.figure(figsize=(20, 10)) |
| --- | --- |


## --- [Page 35] ---


|  | Blocking Matching 실습 |  |
| --- | --- | --- |


| # 왼쪽원본이미지 plt.subplot(1, 2, 1) plt.title("Original Left Image") plt.imshow(cv2.cvtColor(imgL color, cv2.COLOR BGR2RGB)) _ _ plt.axis('off') # 원본크기의Disparity Map plt.subplot(1, 2, 2) plt.title(f"Full-Res SAD Disparity Map\n(NumDisp: {num disp}, Block: {block size})") _ _ plt.imshow(disparity vis, cmap='jet') _ plt.colorbar(fraction=0.046, pad=0.04) plt.axis('off') plt.tight layout() _ plt.show() |  |
| --- | --- |


## --- [Page 36] ---
• 2.5D 영상
• 2D 영상+ 각픽셀의깊이(Depth) 정보가결합된형태의영상

• 2D, 3D 영상과의비교
• 2D 영상: 각픽셀이색상정보(R, G, B)만가짐
• 3D 영상: 사물의모든면(앞, 뒤, 옆)에대한좌표(X, Y, Z) 및색상정보가있어자유로운회전이가능
• 2.5D 영상: 카메라가바라보는시점에서의앞면정보만존재하되, 각픽셀이카메라로부터얼마나떨
어져있는지(깊이값)를알고있는상태

2.5D 영상의개념


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 37] ---
• .ply

• 스탠퍼드대학의그래픽스연구실에서3차원스캐너의데이터를효율적으로저장하기위해개발한포맷

• 각정점(Vertex)에더많은속성(Property)을부여하는데특화되어있음
• 예시)
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0.0 1.0 0.0 255 0 0     # 1번정점: 좌표(0,1,0), 색상(빨강)
-1.0 -1.0 0.0 0 255 0   # 2번정점: 좌표(-1,-1,0), 색상(초록)
1.0 -1.0 0.0 0 0 255    # 3번정점: 좌표(1,-1,0), 색상(파랑)
3 0 1 2                 # 면: 3개의정점을사용하며, 인덱스는0, 1, 2


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 38] ---


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


| import cv2 import numpy as np import matplotlib.pyplot as plt from mpl toolkits.mplot3d import Axes3D _ # 1. 이전단계에서생성한데이터준비(imgL color, disparity float 사용) _ _ scale = 1.0 img small = cv2.resize(imgL color, None, fx=scale, fy=scale) _ _ disp small = cv2.resize(disparity float, None, fx=scale, fy=scale) _ _ h, w = disp small.shape _ # 초점거리(f)와베이스라인(B)은임의의값을설정 f = 0.8 * w # 가상의초점거리 B = 1.0 # 가상의카메라간격 # 3. 3D 좌표계산 x coords, y coords = np.meshgrid(np.arange(w), np.arange(h)) _ _ mask = disp small > 0.5 _ | # Z = (f * B) / disparity z = np.zeros like(disp small) _ _ z[mask] = (f * B) / disp small[mask] _ # X, Y 좌표역투영 x = (x coords - w/2) * z / f _ y = (y coords - h/2) * z / f _ # 4. 시각화데이터준비(유효한점만필터링) points x = x[mask].ravel() _ points y = y[mask].ravel() _ points z = z[mask].ravel() _ # 색상정보추출(BGR -> RGB 및0~1 정규화) colors = cv2.cvtColor(img small, cv2.COLOR BGR2RGB)[mask].reshape(-1, 3) / _ _ 255.0 |
| --- | --- |


## --- [Page 39] ---


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


| def save point cloud ply(filename, points x, points y, points z, colors): _ _ _ _ _ _ """ 포인트클라우드데이터를PLY 파일로저장 points x, y, z: 1차원넘파이배열(좌표) _ colors: 1차원넘파이배열(0~1 범위의RGB 값) """ # 색상을0~255 범위의정수로변환 colors int = (colors * 255).astype(np.uint8) _ # 헤더작성 num points = len(points x) _ _ header = f"""ply format ascii 1.0 element vertex {num points} _ property float x property float y property float z property uchar red property uchar green property uchar blue end header _ """ | # 데이터결합및저장 with open(filename, 'w') as f: f.write(header) for i in range(num points): _ # 좌표와색상을한줄씩기록 f.write(f"{points x[i]} {points y[i]} {points z[i]} {colors int[i, 0]} _ _ _ _ {colors int[i, 1]} {colors int[i, 2]}\n") _ _ print(f"성공: '{filename}' 파일이저장되었습니다. (총{num points}개의점)") _ # 파일저장실행 save point cloud ply("aloe 2 5d.ply", points x, points y, points z, colors) _ _ _ _ _ _ _ _ |
| --- | --- |


## --- [Page 40] ---
• Meshlab
• 메쉬(Triangular Meshes)와Point Cloud와같은3D 모델데이터를편집, 정리, 복원및렌더링
하기위해개발된오픈소스소프트웨어
• Ply, obj 포맷데이터확인때많이사용
• 다운로드: https://www.meshlab.net/


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


| 장점 | 단점 |
| --- | --- |
| 무료 및 오픈 소스: 누구나 제한 없이 사용 가 능. | 메뉴가 비교적 복잡 |
| 강력한 필터: 수백 개의 수학적 필터 제공. | 대용량 데이터 처리 시 가끔 튕김. |
| 광범위한 포맷: PLY, STL, OFF, OBJ, 3DS 등 거의 모든 포맷 지원. | 직접 모델링은 불가 |

## --- [Page 41] ---
• 2.5D 영상(컬러영상+ disparity map) 예시


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


## --- [Page 42] ---
• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지E(d)를최소화하는방식으로깊이지도생성
• 깊이지도를찾는문제를최적화문제로모델링
• 전체에너지E(d)는아래와같이모델링됨

• Edata(d) (Data Term) : 왼쪽영상의픽셀과오른쪽영상의대응픽셀이얼마나유사한지를측정(유사도)
• Esmooth(d) (Smoothness Term) : 인접한픽셀끼리는d값이비슷해야한다는제약조건을부여함.


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 43] ---
• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지E(d)를최소화하는방식으로깊이지도생성
• 깊이지도를찾는문제를최적화문제로모델링
• 전체에너지E(d)는아래와같이모델링됨

• Edata(d) (Data Term) : 왼쪽영상의픽셀과오른쪽영상의대응픽셀이얼마나유사한지를측정(유사도)
• 즉, 왼쪽영상의픽셀과오른쪽영상의대응픽셀간유사도가가장클때Edata(d) 는최소가됨
• 일반적으로블록매칭결과와같음
• 예) Edata(d) = ∑
𝑆𝐴𝐷(𝑥, 𝑦, 𝑑)
(௫,௬)
• Edata(d) (Data Term) 만있다면블록매칭과동일한문제발생

d : 깊이지도에서의disparity(양안시차)들


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 44] ---
• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지E(d)를최소화하는방식으로깊이지도생성
• 깊이지도를찾는문제를최적화문제로모델링
• 전체에너지E(d)는아래와같이모델링됨

• Esmooth(d) (Smoothness Term) : 인접한픽셀끼리는d값이비슷해야한다는제약조건을부여함. 
• 노이즈를억제하고매끄러운깊이지도를만듦

텍스처부족영역
(Textureless Region)

반복패턴
(Repetitive Patterns)
매끄러운깊이값을가짐


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 45] ---
• Global matching(전역매칭) 기반방법
• 깊이지도(disparity map, depth map)의특징
• 깊이가변하는지역을제외하고주변픽셀의d값이서로비슷함(비슷한깊이를가지기때문)

Left image
Right image

Disparity map of 
left image

Disparity map of 
right image


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 46] ---
• Global matching(전역매칭) 기반방법
• 깊이지도(disparity map, depth map)의특징
• 깊이가변하는지역을제외하고주변픽셀의d값이서로비슷함(비슷한깊이를가지기때문)

Left image
Right image

Disparity map of 
left image

Disparity map of 
right image


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 47] ---
• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지E(d)를최소화하는방식으로깊이지도생성
• 깊이지도를찾는문제를최적화문제로모델링
• 전체에너지E(d)는아래와같이모델링됨

• Esmooth(d) (Smoothness Term) : 인접한픽셀끼리는d값이비슷해야한다는제약조건을부여함. 
• 즉, 인접한픽셀간d값의차가작을수록Esmooth(d) (Smoothness Term)는작아짐
• 만약깊이지도의모든d가동일하면Esmooth(d) = 0 (최소) 가됨
• 예) 인접한두픽셀p, q의disparity를dp, dq라고하면

: p, q의disparity가다르면무조건λ만큼에너지가증가

: p, q의disparity의차가클수록에너지가더많이증가

1)

2)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 48] ---
• Global matching(전역매칭) 기반방법
• 영상의깊이값들의전체에너지E(d)를최소화하는방식으로깊이지도생성
• 깊이지도를찾는문제를최적화문제로모델링
• 전체에너지E(d)는아래와같이모델링됨

• Edata(d)와Esmooth(d) 간에는trade-off 관계가있음
• E(d)는최소화하는것은Edata(d)와Esmooth(d) 간의최적의균형을찾는것
• E(d)를최소화하는최적의깊이지도dopt를찾는것은현실적으로불가능하기때문에다양한최적화방
법으로dopt 의근사해를구함


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 49] ---
• Global matching(전역매칭) 기반방법
• E(d)를최소화하는d의근사해를구하는대표적인방법
• GraphCut
• 깊이지도를픽셀간그래프로모델링하고2차원최적화를수행
• Dynamic programming 
• 각라인별로매칭비용에대한테이블을만들고DP로최적해를구함(1차원최적화)

Stereo Image

(Left image)

Ground truth (정답)

disparity map

Block matching

(SAD)

Global matching

(GraphCut)

Global matching 
(Dynamic Programming)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 50] ---
• Global matching(전역매칭) 기반방법
• E(d)를최소화하는d의근사해를구하는대표적인방법
• Dynamic programming 
• 각라인별로매칭비용에대한테이블을만들고DP로최적해를구함(1차원최적화)
• 예) 영상의크기가10x8이고search range가5라고하면

Search range


|  | Disparity Estimation |  |
| --- | --- | --- |


| SAD(0, 0) |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | SAD(x, d) |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  | SAD(9, 4) |

## --- [Page 51] ---
• Global matching(전역매칭) 기반방법
• E(d)를최소화하는d의근사해를구하는대표적인방법
• Dynamic programming 
• 각라인별로매칭비용에대한테이블을만들고DP로최적해를구함(1차원최적화)
• 예) 영상의크기가10x8이고search range가5라고하면

Search range

8개의테이블에대해DP 수행


|  | Disparity Estimation |  |
| --- | --- | --- |


| SAD(0, 0) |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | SAD(x, d) |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  | SAD(9, 4) |

## --- [Page 52] ---
• Global matching(전역매칭) 기반방법
• E(d)를최소화하는d의근사해를구하는대표적인방법
• GraphCut
• 영상의크기가WxH이고search range가D라고하면WxHxD의cost volume을생성

…

W

H

D

cost volume의예

SAD(x, y, d)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 53] ---
• Global matching(전역매칭) 기반방법
• E(d)를최소화하는d의근사해를구하는대표적인방법
• GraphCut
• 영상의크기가WxH이고search range가D라고하면WxHxD의cost volume을생성
• Cost volume을그래프로모델링하고minimum cut을구함으로써2차원최적화를수행


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 54] ---
1. 과제#1 이후부터현재까지실습코드를모두수행하고결과출력(2점)

• 각수행마다코드에는어떤과정인지주석처리

제출방법: LMS로압축파일제출

• 코랩에서다음과같은이름을가진노트북파일을생성한다.                                                                      
ComputerVision_2ndHW_Class본인소속반_학번_영문이름. ipynb

예)  ComputerVision_2ndHW_Class1_24xxxxx_HanshinLim.ipynb

• 생성된코랩노트북파일(.ipynb)에서코드를구현및결과를출력한다. 결과출력시어떤결과인지를명시
한다.

• Due :

• 1반: 5월11일밤12시(이후제출0점)

• 2반: 5월13일밤12시(이후제출0점)


|  | 과제 #2 |  |
| --- | --- | --- |


## --- [Page 55] ---
• Semi-global matching(세미-전역매칭) 기반방법
• 블록매칭의속도와전역매칭의정확도사이에서균형을맞춘방식
• Dynamic programming 기반전역매칭방법은한라인에대해서만최적화를수행하므로라인간깊이불
일치로인한줄무늬가생김
•
Semi-global matching에서는8방향또는16방향에서Dynamic programming을수행하여각각의방향에
대해경로비용을계산
• 이후모든방향의경로비용을더하고이로부터최적의깊이지도를찾음


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 56] ---
• Semi-global matching(세미-전역매칭) 기반방법
• 블록매칭의속도와전역매칭의정확도사이에서균형을맞춘방식
• Dynamic programming 기반전역매칭방법은한라인에대해서만최적화를수행하므로라인간깊이불
일치로인한줄무늬가생김
•
Semi-global matching에서는8방향또는16방향에서Dynamic programming을수행하여각각의방향에
대해경로비용을계산
• 이후모든방향의경로비용을더하고이로부터최적의깊이지도를찾음

Smoothness term의예:


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 57] ---
• Semi-global matching(세미-전역매칭) 기반방법
• 블록매칭의속도와전역매칭의정확도사이에서균형을맞춘방식
• 장점
1.
여러방향의정보를취합하므로기존Dynamic Programming의문제인가로줄무늬가거의나타나지
않음
2.
각경로계산이독립적이어서GPU나FPGA를통한병렬화에매우유리


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 58] ---


|  | Semi-Global Matching 실습 |  |
| --- | --- | --- |


| import cv2 import numpy as np from matplotlib import pyplot as plt import urllib.request import time url1 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg' url2 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg' urllib.request.urlretrieve(url1, 'aloeL.jpg') urllib.request.urlretrieve(url2, 'aloeR.jpg') imgL = cv2.imread('aloeL.jpg') imgR = cv2.imread('aloeR.jpg') # 2. StereoSGBM 파라미터설정 window size = 3 _ min disp = 0 _ num disp = 16 * 8 _ # 모드변경을통한속도비교 # cv2.STEREO SGBM MODE SGBM 3WAY (가장빠름) _ _ _ _ # cv2.STEREO SGBM MODE SGBM (기본, 5방향) _ _ _ # cv2.STEREO SGBM MODE HH (8방향, 가장느리지만정밀) _ _ _ | current mode = cv2.STEREO SGBM MODE SGBM _ _ _ _ stereo = cv2.StereoSGBM create( _ minDisparity= min disp, _ numDisparities=num disp, _ blockSize=window size, _ P1=8 * 3 * window size**2, _ P2=32 * 3 * window size**2, _ disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, preFilterCap=63, mode=current mode _ ) # 3. 속도측정시작 start time = time.time() # 시작시각기록 _ # 깊이지도계산 disparity raw = stereo.compute(imgL, imgR) _ |
| --- | --- |


## --- [Page 59] ---


|  | Semi-Global Matching 실습 |  |
| --- | --- | --- |


| # 속도측정종료 end time = time.time() # 종료시각기록 _ execution time = (end time - start time) * 1000 # ms 단위로변환 _ _ _ # disparity 정규화 disparity = disparity raw.astype(np.float32) / 16.0 _ # 4. 결과출력및시각화 print(f"SGBM Execution Time: {execution time:.2f} ms") _ plt.figure(figsize=(14, 7)) plt.subplot(121) plt.imshow(cv2.cvtColor(imgL, cv2.COLOR BGR2RGB)) _ plt.title('Left Image') plt.axis('off') plt.subplot(122) plt.imshow(disparity, 'jet') plt.title(f'Disparity Map (SGBM)\nTime: {execution time:.2f} ms') _ plt.axis('off') plt.tight layout() _ plt.show() | ㅁ |
| --- | --- |


## --- [Page 60] ---
• 블록매칭
1. 기준블록(왼쪽영상)과비교블록(오른쪽영상)에서생성된비트열끼리hamming distance를구함
2. Hamming distance가가장작은비교블록을매칭블록으로판정
예)

Census Transform
11000001

Census Transform

비교블록

11000000

11000001
XOR
11000000

00000001         Hamming distance : 1


|  | Disparity Estimation |  |
| --- | --- | --- |


| 97 | 94 | 43 |
| --- | --- | --- |
| 89 | 85 | 22 |
| 77 | 30 | 15 |

| 78 | 75 | 23 |
| --- | --- | --- |
| 63 | 65 | 12 |
| 57 | 14 | 5 |

## --- [Page 61] ---
• Census Transform
• Census transform의장점
• 밝기변화의전체적인패턴(Hamming Distance)은크게변하지않아결과가안정적임
• 주변보다밝은지어두운지의순서(비트열)를기록함으로써매칭에구조정보를사용
• 일반적인연산보다훨씬적은클럭사이클로매칭비용을계산할수있고GPU 등의병렬
처리에적합


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 62] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
1. 특징추출(Feature extraction)
2. Cost volume 생성
3. 비용최적화(비용집계, cost aggregation)
4. Disparity 추정


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 63] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• Deep learning 기반스테레오매칭알고리즘(RAFT-Stereo)의예

RAFT-Stereo 전체구조
RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching, 
3DV,  2021.


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 64] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• Deep learning 기반스테레오매칭알고리즘(RAFT-Stereo)의예

RAFT-Stereo 전체구조

1. 특징추출
(Feature extraction)

2. Cost volume 생성

3. 비용최적화
(비용집계, cost aggregation)

4. Disparity 추정

RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching, 
3DV,  2021.


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 65] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
1. 특징추출(Feature extraction)
• 좌우이미지간의유사도(Correlation)를계산하기위한특징지도를만듦
• CNN Encoder 등을사용하여특징추출
• 예) RAFT-Stereo의특징추출기(feature encoder)

RAFT-Stereo의특징추출기(feature encoder)

Left or Right
image

Feature / Context 
Encoder

특징지도
(Feature maps)

좌우각각
H/4 x W/4 x 256


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 66] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성
• Cost volume : 각픽셀과disparity에대한매칭비용(matching cost)을모아둔데이터구조
• 초기cost volume의매칭비용은픽셀하나하나의독립적인정보만담고있어노이즈가많고잘못
된매칭을포함하고있음

…

W

H

D

cost volume의예

매칭비용
(예) SAD(x, y, d))


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 67] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성(RAFT-Stereo의예)
• 좌측영상의한행(Row)에있는모든특징벡터와우측영상의같은행에있는모든특징벡터간의
내적(Dot Product)을구함. 즉,

• 결과적으로W/4 xH/4 xW/4 크기의텐서가생성
• 마지막차원W/4 에대해커널크기1, 2, 4, 8 등으로Average Pooling을수행하여4개층의cost
volume을만듦

: feature encoder 출력

RAFT-Stereo의cost volume(correlation pyramid, 여기서H, W는실제로는입력에서/4 )


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 68] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성(RAFT-Stereo의예)
• 좌측영상의한행(Row)에있는모든특징벡터와우측영상의같은행에있는모든특징벡터간의
내적(Dot Product)을구함. 즉,
• 결과적으로W/4 x H/4 x W/4 크기의텐서가생성

W/4

왼쪽영상의

특징지도

오른쪽영상의

특징지도

256채널

H/4

W/4

H/4


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 69] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성(RAFT-Stereo의예)
• 좌측영상의한행(Row)에있는모든특징벡터와우측영상의같은행에있는모든특징벡터간의
내적(Dot Product)을구함.
• 결과적으로W/4 x H/4 x W/4 크기의텐서가생성

W/4

왼쪽영상의

특징지도

오른쪽영상의

특징지도

H/4

256채널
256채널

W/4

H/4

W/4

H/4

W/4 x H/4 x W/4 크기의
Cost volume(Correlation pyramid)

W/4


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 70] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성(RAFT-Stereo의예)
• 마지막차원W/4 에대해커널크기1, 2, 4, 8 등으로Average Pooling을수행하여4개층의cost
volume을만듦
• 해상도가큰cost volume은텍스처가많은미세한영역의탐색에유리
• 해상도가작은cost volume은텍스처가부족하거나넓은영역의탐색에유리

RAFT-Stereo의cost volume(correlation pyramid, 여기서H, W는실제로는입력에서/4 )


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 71] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
2. Cost volume 생성(RAFT-Stereo의예)
• 마지막차원W/4 에대해커널크기1, 2, 4, 8 등으로Average Pooling을수행하여4개층의cost
volume을만듦
• 해상도가큰cost volume은텍스처가많은미세한영역의탐색에유리
• 해상도가작은cost volume은텍스처가부족하거나넓은영역의탐색에유리


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 72] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
3. 비용최적화(비용집계, cost aggregation)
• 초기cost volume을3D CNN 또는반복적GRU 업데이트(RAFT-Stereo) 등으로비용을정제

3. 비용최적화
(비용집계, cost aggregation)

GRU(Gated Recurrent Unit)


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 73] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• 일반적으로아래의4단계로구성됨
4. Disparity 추정
• Cost volume에서정제된비용(Cost)을확률분포로변환한뒤모든disparity의후보들에대해확률
만큼가중평균하여구함(기대값)
• RAFT-Stereo의경우GRU가예측한변화량(Δd)을기존disparity에계속더해서최종값을구함

4. Disparity 추정


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 74] ---
• Deep learning(딥러닝) 기반방법
• 두영상으로부터특징맵을만들고이로부터영상의깊이(양안시차)를추정
• Deep learning 기반스테레오매칭알고리즘(RAFT-Stereo)의예

RAFT-Stereo 전체구조

1. 특징추출
(Feature extraction)

2. Cost volume 생성

3. 비용최적화
(비용집계, cost aggregation)

4. Disparity 추정

RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching, 
3DV,  2021.


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 75] ---


|  | RAFT-Stereo 실습 |  |
| --- | --- | --- |


| [수정] -> [노트북설정] -> [T4 GPU] 활성화 1. 환경설정및저장소클론 import os # RAFT-Stereo 공식repository 클론 !git clone https://github.com/princeton-vl/RAFT-Stereo.git %cd RAFT-Stereo # 필요한라이브러리설치 !pip install coremltools # 모델구조에따라필요할수있음 2. Middlebury 데이터셋으로사전학습된모델(Checkpoints) 다운로드 # 공식모델다운로드스크립트실행 !bash download models.sh _ # 다운로드가완료되면models/ 폴더내에체크포인트들이생성됨 | ㅁ |
| --- | --- |


## --- [Page 76] ---


|  | RAFT-Stereo 실습 |  |
| --- | --- | --- |


| import sys sys.path.append(‘.') import argparse import time import cv2 import numpy as np import torch from PIL import Image from matplotlib import pyplot as plt import urllib.request From core.utils.utils import InputPadder from core.raft stereo import RAFTStereo _ # 모델로드함수 def load model(restore ckpt): _ _ # 기본설정(기본값사용) parser = argparse.ArgumentParser() args = parser.parse known args()[0] _ _ args.restore ckpt = restore ckpt _ _ args.mixed precision = False _ args.slow fast gru = False _ _ | args.valid iters = 32 # 반복횟수(정밀도조절) _ args.hidden dims = [128]*3 _ args.context dims = [128]*3 _ args.corr implementation = "reg" _ args.shared backbone = False _ args.corr levels = 4 _ args.corr radius = 4 _ args.n downsample = 2 _ args.n gru layers = 3 _ _ args.context norm = 'batch' _ args.inter num = 2 _ model = torch.nn.DataParallel(RAFTStereo(args)) model.load state dict(torch.load(args.restore ckpt)) _ _ _ model = model.module model.cuda() model.eval() return model |
| --- | --- |


## --- [Page 77] ---


|  | RAFT-Stereo 실습 |  |
| --- | --- | --- |


| # 4. 이미지준비및추론 model = load model('models/raftstereo-middlebury.pth') _ url1 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg' url2 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg' urllib.request.urlretrieve(url1, 'aloeL.jpg') urllib.request.urlretrieve(url2, 'aloeR.jpg') imgL = cv2.imread('aloeL.jpg') imgR = cv2.imread('aloeR.jpg') image1 = torch.from numpy(imgL).permute(2, 0, 1).float()[None].cuda() _ image2 = torch.from numpy(imgR).permute(2, 0, 1).float()[None].cuda() _ # 이미지패딩(모델입력크기최적화) padder = InputPadder(image1.shape) image1, image2 = padder.pad(image1, image2) with torch.no grad(): _ start time = time.time() _ # 32회반복업데이트수행 , flow up = model(image1, image2, iters=32, test mode=True) _ _ _ end time = time.time() _ | # 결과복원(시차값은Flow의음수값으로표현됨) disparity = -flow up.cpu().numpy().squeeze() _ execution time = (end time - start time) * 1000 _ _ _ # 5. 시각화 from mpl toolkits.axes grid1 import make axes locatable _ _ _ _ fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) ax1.imshow(imgL) ax1.set title('Original Left Image') _ ax1.axis('off') div1 = make axes locatable(ax1) _ _ div1.append axes("right", size="5%", pad=0.1).axis('off') _ im = ax2.imshow(disparity, cmap='jet') ax2.set title(f'RAFT-Stereo Disparity\nTime: {execution time:.2f} ms') _ _ ax2.axis('off') div2 = make axes locatable(ax2) _ _ cax = div2.append axes("right", size="5%", pad=0.1) _ plt.colorbar(im, cax=cax) plt.show() |
| --- | --- |


## --- [Page 78] ---


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


| def save point cloud as 25d(filename, img bgr, disp): _ _ _ _ _ h, w = disp.shape if img bgr.shape[0] != h or img bgr.shape[1] != w: _ _ img bgr = cv2.resize(img bgr, (w, h), interpolation=cv2.INTER LINEAR) _ _ _ # 카메라파라미터설정 f = 0.8 * w # 초점거리 B = 1.0 # 베이스라인 # 3D 좌표계산 x coords, y coords = np.meshgrid(np.arange(w), np.arange(h)) _ _ # 0 이하및노이즈제거 mask = disp > 0.1 # Depth 계산: Z = (f * B) / disparity z = (f * B) / (disp + 1e-6) x = (x coords - w/2) * z / f _ y = (y coords - h/2) * z / f _ # 3D 정렬을위한좌표축보정 points = np.stack([x[mask], -y[mask], -z[mask]], axis=1) | # 색상정보추출(BGR -> RGB) colors = cv2.cvtColor(img bgr, cv2.COLOR BGR2RGB)[mask] _ _ # PLY 파일헤더및데이터쓰기 num points = len(points) _ header = ( f"ply\n" f"format ascii 1.0\n" f"element vertex {num points}\n" _ f"property float x\n" f"property float y\n" f"property float z\n" f"property uchar red\n" f"property uchar green\n" f"property uchar blue\n" f"end header\n" _ ) |
| --- | --- |


## --- [Page 79] ---


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


| with open(filename, 'w') as f out: _ f out.write(header) _ # 좌표(fmt %.4f)와색상(fmt %d)을결합하여저장 np.savetxt(f out, np.column stack((points, colors)), _ _ fmt='%.4f %.4f %.4f %d %d %d') print(f"\n[성공] 2.5D 영상파일저장완료: {filename}") print(f"총포인트수: {num points:,}개") _ # 2.5D PLY 저장 save point cloud as 25d("aloe 2 5d result.ply", imgL, disparity) _ _ _ _ _ _ _ |  |
| --- | --- |


## --- [Page 80] ---
RAFT-Stereo로구한깊이지도로부터2.5D 영상생성결과


|  | 2.5D 영상 실습 |  |
| --- | --- | --- |


## --- [Page 81] ---
• Deep learning(딥러닝) 기반방법
• Deep learning 기반방법의장단점
• 장점
• 텍스처부족, 텍스처반복영역등에서높은정확도
• 조명변화에도비교적강인함

• 단점
• 일반화(generalization) 문제
• 학습데이터가필요
• 비교적높은컴퓨팅파워(GPU, VRAM) 필요


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 82] ---
• Deep learning(딥러닝) 기반방법
• Deep learning 기반방법의학습
• 손실함수(Loss function)의예
• Ground truth(정답데이터, dgt)가존재하는지도학습의경우
• L1 Loss

• N : 유효한(GT가존재하는) disparity의수
• dpred : predicted disparity

• Smooth L1 (Huber Loss)

• 오차가작을때는L2 loss처럼동작
• 오차가클때는L1 loss처럼동작

x = dgt - dpred

Huber loss

L2 loss


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 83] ---
• Deep learning(딥러닝) 기반방법
• Deep learning 기반방법의학습
• 손실함수(Loss function)의예
• Ground truth(정답데이터, dgt)가존재하는지도학습의경우
• RAFT-Stereo의Loss(Sequential loss)

• T : 총stage의수
• dt : t번째stage에서예측된disparity
• 매stage마다L1 loss를누적하여계산
• 나중의stage의예측값에더높은가중치를부여

𝛾= 0.9


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 84] ---
• Deep learning(딥러닝) 기반방법
• Deep learning 기반방법의학습
• 주요데이터셋
• Middlebury stereo
• https://vision.middlebury.edu/stereo/data/
• 비교적고해상도의데이터셋을제공
(Full resolution 기준약3000 x 2000)
• 동일한장면에대해서로다른조명
(Lighting)과노출(Exposure) 조건을제공
• KITTI
• https://www.cvlibs.net/datasets/kitti/
• 실제도로주행환경에서의데이터셋
• 해상도는약1242 x 375 (가로로긴형태)
• 실제야외촬영에서발생하는다양한변
수가포함되어있음

Middlebury stereo 2014 데이터셋예

KITTI stereo 2015 데이터셋예


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 85] ---
• Deep learning(딥러닝) 기반방법
• Deep learning 기반방법의학습
• 주요데이터셋
• ETH3D
• https://www.eth3d.net/
• 실내부터야외까지다양한상황에서촬
영한데이터셋
• 스테레오데이터셋의경우약940x490
의비교적저해상도

ETH3D 2시점스테레오데이터셋예


|  | Disparity Estimation |  |
| --- | --- | --- |


## --- [Page 86] ---
• 핀홀카메라모델의기본구조

• 스테레오영상의기하구조

• 스테레오영상으로부터깊이추정방법


|  | 내용 정리 |  |
| --- | --- | --- |


## --- [Page 87] ---
• 카메라파라미터들과3D 기하의기본

• 3D 구조복원SW(Colmap) 기본내용및실습


|  | 다음 주 강의 내용 |  |
| --- | --- | --- |
