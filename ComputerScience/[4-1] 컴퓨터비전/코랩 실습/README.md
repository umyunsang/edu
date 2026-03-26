# 컴퓨터비전 코랩 실습 목록

이 폴더는 컴퓨터비전 강의의 주요 알고리즘과 기법들을 Google Colab 환경에서 실습한 노트북 파일들을 담고 있습니다.

## 실습 항목

### 1. 영상 기초 및 변환
| 실습 명칭 | 주요 내용 | 파일 링크 |
| :--- | :--- | :--- |
| **RGB to YUV 변환** | 컬러 모델 간의 변환 및 채널별 특징 확인 | [노트북](./RGB_to_YUV_변환.ipynb) |
| **감마 보정 (Gamma Correction)** | 영상의 밝기 및 대비 조절을 위한 비선형 보정 | [노트북](./Gamma_Correction.ipynb) |
| **기하 변환 (Geometric Transform)** | 영상의 이동, 회전, 대칭 등 기하학적 변형 | [노트북](./Geometric_Transform_기하변환.ipynb) |
| **보간법 (Interpolation)** | 영상 확대/축소 시 픽셀 값을 채우는 기법 | [노트북](./Interpolation_보간.ipynb) |
| **히스토그램 평활화** | 명암 분포를 균일하게 만들어 대비를 향상시키는 기법 | [노트북](./히스토그램_평활화.ipynb) |

### 2. 필터링 및 에지 검출
| 실습 명칭 | 주요 내용 | 파일 링크 |
| :--- | :--- | :--- |
| **컨벌루션 필터링** | 다양한 커널을 이용한 영상의 컨벌루션 연산 기초 | [노트북](./Convolution_Filtering.ipynb) |
| **가우시안 필터링** | 가우시안 분포를 이용한 영상 블러링 및 잡음 제거 | [노트북](./Gaussian_Filtering.ipynb) |
| **소벨 에지 검출** | Sobel 커널을 이용한 수평/수직 에지 추출 | [노트북](./Sobel_Edge_Detection.ipynb) |
| **캐니 에지 검출** | Canny 알고리즘을 이용한 정교한 에지 검출 | [노트북](./Canny_Edge_Detection.ipynb) |
| **모폴로지 연산** | 침식, 팽창 등을 이용한 형태학적 영상 처리 | [노트북](./Morphology_모폴로지.ipynb) |

### 3. 특징 추출 및 영상 분할
| 실습 명칭 | 주요 내용 | 파일 링크 |
| :--- | :--- | :--- |
| **해리스 코너 검출** | Harris Corner Detection을 이용한 특징점 추출 | [노트북](./해리스%20코너%20검출%20실습.ipynb) |
| **슈퍼픽셀 분할** | SLIC 알고리즘을 이용한 슈퍼픽셀 단위 분할 | [노트북](./슈퍼픽셀%20분할%20실습.ipynb) |
| **최적화 분할** | SLIC + N-Cut을 결합한 최적 영역 병합 및 분할 | [노트북](./최적화%20분할%20실습.ipynb) |
| **영상 품질 측정** | PSNR 및 SSIM 지표를 활용한 화질 정량 측정 | [노트북](./영상%20품질%20측정%20실습.ipynb) |

## 실습 환경
* **Language**: Python 3.10+
* **Library**: OpenCV, NumPy, Scikit-image, Matplotlib
* **Platform**: Google Colab (추천)
