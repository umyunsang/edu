### [Page 1]
1/32
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능


|  |  |
| --- | --- |
|  |  |
|  | 1/32 |


### [Page 2]
2/32
Contents
1. CNN 개요
2. Convolution 연산
3. 3D 데이터의Convolution 연산



### [Page 3]
3/32
Contents
1. CNN 개요
2. Convolution 연산
3. 3D 데이터의Convolution 연산



### [Page 4]
4/32
Convolutional Neural Networks - 개요

MLP 모델의문제점
•
평탄화된이미지를입력으로받아공간적(형상) 정보가사라짐
•
CNN에비해필요한weight parameter의개수가많음
MLP의입출력구조예시
평탄화
MLP
예측결과
28 x 28
784 x 1
10 x 1



### [Page 5]
Fully Connected (FC) Layer
Convolutional neural network (CNN)

FC Layer의문제점
•
FC Layer는1차원데이터로입력을받음
•
따라서, 2D 이미지데이터를1D로평탄화해입력함이미지의공간적(형상) 정보가무시됨
3072
1
10 x 3072
weights
activation
input
1
10
ex) 32x32x3 image -> stretch to 3072 x 1
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5



### [Page 6]
Overview
Convolutional neural network (CNN)
•
이미지의특성에따라뉴런이다르게활성화됨
Cat image by CNX OpenStax is licensed  
under CC BY 4.0; changes made



### [Page 7]
Overview
Convolutional neural network (CNN)
•
2D이미지를Input으로사용



### [Page 8]
ImageNet Large Scale Visual Recognition Challenge (ILSVRC)



### [Page 9]
ImageNet Large Scale Visual Recognition Challenge (ILSVRC)



### [Page 10]
10/32
Convolutional Neural Networks - 개요

Perceptron 모델의문제점
•
평탄화된이미지를입력으로받아공간적(형상) 정보가사라짐
•
CNN에비해필요한weight parameter의개수가많음
Fully connected layer
(#weights: 7,840)
5x5 Convolution layer
(#weights: 1,600)
[5x5x1]x64
32
32
28
28
64
784x10



### [Page 11]
11/32
Convolutional Neural Networks - 개요

CNN 모델개요
•
합성곱연산(Convolution)을수행해이미지의특징(Feature) 추출
•
이미지를입력으로받는분야에서높은성능을보임(Ex. Image classification, Super-resolution, Denoising)
Input image
3 (Channel)
32 (Height)
32 (Width)
Filter
(Kernel)
3x3x3 filters
32 (Channel)
30 (Height)
30 (Width)
…
filter 1
filter 2
filter 32
Feature maps
*
(Conv.)



### [Page 12]
12/14
[AI 민주화] 누구나AI로부터차별받지않고AI 기술에쉽게접근하고개발할수있도록하는것!

Democratization of AI



### [Page 13]
13/32
Contents
1. CNN 개요
2. Convolution 연산
3. 3D 데이터의Convolution 연산



### [Page 14]
14/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
2
0
1
0
1
2
1
0
2
4x4x1 image
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
3x3x1 filter
Input image
Filter
(Kernel)


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |


### [Page 15]
15/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
2
0
1
0
1
2
1
0
2
4x4x1 image
Input image
Filter
(Kernel)
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
3x3x1 filter
입력이미지와필터의채널은같아야함


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |


### [Page 16]
16/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
Input image
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
2
0
1
0
1
2
1
0
2
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
Conv.
연산
*
Input image
Filter (Kernel)
Output = 1x2 + 2x0 + 3x1 + 0x0 + 1x1 + 2x2 + 3x1 + 0x0 + 1x2
= 15
15


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 15 |  |
| --- | --- |
|  |  |


### [Page 17]
17/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
Input image
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
2
0
1
0
1
2
1
0
2
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
Conv.
연산
*
Input image
Filter (Kernel)
Input image
Input image
Input image


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |


### [Page 18]
18/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
2
0
1
0
1
2
1
0
2
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
Conv.
연산
*
Input image
Filter (Kernel)
15
15 16
15 16
6
15 16
6
15


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 15 |  |
| --- | --- |
|  |  |

| 15 | 16 |
| --- | --- |
|  |  |

| 15 | 16 |
| --- | --- |
| 6 |  |

| 15 | 16 |
| --- | --- |
| 6 | 15 |


### [Page 19]
19/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
2
0
1
0
1
2
1
0
2
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
Conv.
연산
*
Input image
Filter (Kernel)
15 16
6
15
Output feature map


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 16 |
| --- | --- |
| 6 | 15 |


### [Page 20]
20/32
Convolutional Neural Networks - Convolution 연산

2D 이미지입력에대한convolution 연산예시
Output feature map
Input image



### [Page 21]
21/32
Convolutional Neural Networks - Convolution 연산

Stride: Convolution 연산의step size
•
Stride가커질수록feature map의크기는작아짐
15 16
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
0
1
0
1
2
1
0
2
*
2
0
1
0
1
2
1
0
2
*
Stride=1
15
Input image
Filter
Output
feature map


| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 | 16 |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |


### [Page 22]
22/32
Convolutional Neural Networks - Convolution 연산

Stride: Convolution 연산의step size
•
Stride가커질수록feature map의크기는작아짐
15 16
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
0
1
0
1
2
1
0
2
*
2
0
1
0
1
2
1
0
2
*
Stride=1
15
Input image
Filter
Output
feature map
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
3
0
1
2
0
1
0
1
2
1
0
2
*
2
0
1
0
1
2
1
0
2
*
Stride=2
15
15 17
Input image
Filter
Output
feature map


| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 |  |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 | 16 |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 17 |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |


### [Page 23]
23/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
7x7 input (spatially)  
assume 3x3 filter
7
7

Stride
•
Filter가Convolution연산을수행하는Step size



### [Page 24]
24/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
7x7 input (spatially)  
assume 3x3 filter
7
7

Stride
•
Filter가Convolution연산을수행하는Step size



### [Page 25]
25/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
7x7 input (spatially)  
assume 3x3 filter
7
7

Stride
•
Filter가Convolution연산을수행하는Step size



### [Page 26]
26/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
7x7 input (spatially)  
assume 3x3 filter
7
7

Stride
•
Filter가Convolution연산을수행하는Step size



### [Page 27]
27/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
7x7 input (spatially)  
assume 3x3 filter
=> 5x5 output
7
7

Stride
•
Filter가Convolution연산을수행하는Step size
5
5



### [Page 28]
28/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Stride
•
Filter가Convolution연산을수행하는Step size
7x7 input (spatially)  
assume 3x3 filter  ap
plied with stride 2
7
7



### [Page 29]
29/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Stride
•
Filter가Convolution연산을수행하는Step size
7x7 input (spatially)  
assume 3x3 filter  ap
plied with stride 2
7
7



### [Page 30]
30/32
Convolution Layer
Convolutional neural network (CNN)
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Stride
•
Filter가Convolution연산을수행하는Step size
7x7 input (spatially)  
assume 3x3 filter  ap
plied with stride 2
=> 3x3 output!
7
7
3
3



### [Page 31]
31/32
Convolutional Neural Networks - Convolution 연산

Padding: Input image 주변값을특정값(주로0) 으로채워줌
•
Convolution 연산으로boundary 정보가소실되는문제를방지
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
2
0
1
0
1
2
1
0
2
*
Padding size = 0
Input image
Filter
Output
feature map
15 16
6
15
stride: 1


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 16 |
| --- | --- |
| 6 | 15 |


### [Page 32]
32/32
Convolutional Neural Networks - Convolution 연산

Padding: Input image 주변값을특정값(주로0) 으로채워줌
•
Convolution 연산으로boundary 정보가소실되는문제를방지
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
2
0
1
0
1
2
1
0
2
*
Padding size = 0
Input image
Filter
Output
feature map
15 16
6
15
stride: 1
2
0
1
0
1
2
1
0
2
0
0
0
0
0
0
0
1
2
3
0
0
0
0
1
2
3
0
0
3
0
1
2
0
0
2
3
0
1
0
0
0
0
0
0
0
*
Padding size = 1
Input image
Filter
Output
feature map
7
12 10
2
4
15 16 10
10
6
15
6
8
4
7
3
stride: 1


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 16 |
| --- | --- |
| 6 | 15 |

| 0 | 0 | 0 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 0 |
| 0 | 0 | 1 | 2 | 3 | 0 |
| 0 | 3 | 0 | 1 | 2 | 0 |
| 0 | 2 | 3 | 0 | 1 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 |

| 7 | 12 | 10 | 2 |
| --- | --- | --- | --- |
| 4 | 15 | 16 | 10 |
| 10 | 6 | 15 | 6 |
| 8 | 4 | 7 | 3 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |


### [Page 33]
33/32
Convolution Layer
Convolutional neural network (CNN)

Padding
•
Input data의주변을특정값으로채움(보통0을사용)
•
Convolution 연산으로boundary의정보가소실되는문제점을방지



### [Page 34]
34/32
Convolutional Neural Networks - Convolution 연산

Convolution 연산의출력크기계산
•
출력크기는filter size, stride, padding size에따라달라짐
•
출력크기는정수로나누어떨어져야함
2
Output Height
1
H
P
FH
OH
S





2
Output Width
1
W
P
FW
OW
S





(H, W): Input data size
(FW, FH): Filter size
P: Padding size
S: Stride



### [Page 35]
35/32
Convolutional Neural Networks - Convolution 연산

Pooling: Input data의dimension을subsampling 하는방법
2x2 max pooling 예시
2x2 average pooling 예시



### [Page 36]
36/32
Contents
1. CNN 개요
2. Convolution 연산
3. 3D 데이터의Convolution 연산



### [Page 37]
37/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
입력이미지와필터의채널은같아야함
Input image
(Input feature map)
Filter
(Kernel)
4x4x3
3x3x3
*
2x2x1
Output
feature map



### [Page 38]
38/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
Input image
Filter
(Kernel)
4x4x3
3x3x3
*
2x2x1
Output
feature map



### [Page 39]
39/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산
(1)
4x4x3
(Input)
3x3x3 (Filter)



### [Page 40]
40/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산
(1)
(2)
4x4x3
(Input)
3x3x3 (Filter)



### [Page 41]
41/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산
(1)
(2)
(3)
(4)
4x4x3
(Input)
3x3x3 (Filter)
2x2x1
Output
feature map



### [Page 42]
42/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
63
(1)


|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 |  |
| --- | --- |
|  |  |


### [Page 43]
43/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
63
63 55
(1)
(2)


|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  |  | 4 |  | 2 |  | 1 |  | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 |  | 0 |  | 6 |  | 5 |  | 4 |
| 1 |  | 2 |  | 3 |  | 0 |  | 3 | 2 |
| 0 |  | 1 |  | 2 |  | 3 |  | 0 | 5 |
| 3 |  | 0 |  | 1 |  | 2 |  | 1 |  |
| 2 |  | 3 |  | 0 |  | 1 |  |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 |  |
| --- | --- |
|  |  |

| 63 | 55 |
| --- | --- |
|  |  |


### [Page 44]
44/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
63
63 55
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
0
2
0
2
0
1
3
2
0
2
0
1
0
1
2
1
0
2
*
4
2
1
2
4
2
5
3
0
6
5
3
0
1
1
2
3
0
0
1
2
3
3
0
1
2
2
3
0
1
63 55
18
63 55
18 51
Output
feature map
(1)
(2)
(3)
(4)


|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  |  | 4 |  | 2 |  | 1 |  | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 |  | 0 |  | 6 |  | 5 |  | 4 |
| 1 |  | 2 |  | 3 |  | 0 |  | 3 | 2 |
| 0 |  | 1 |  | 2 |  | 3 |  | 0 | 5 |
| 3 |  | 0 |  | 1 |  | 2 |  | 1 |  |
| 2 |  | 3 |  | 0 |  | 1 |  |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 |  |
| --- | --- |
|  |  |

| 63 | 55 |
| --- | --- |
|  |  |

|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 | 55 |
| --- | --- |
| 18 |  |

| 63 | 55 |
| --- | --- |
| 18 | 51 |


### [Page 45]
45/32
Convolutional Neural Networks - 3D 데이터의Convolution 연산

3D 이미지(RGB) 입력에대한2D convolution 연산
•
Output feature map의채널수는filter의개수(N)와같음
Input image
Filter
(Kernel)
4x4x3
3x3x3 filters
*
2x2x1
Output
feature map
…
filter 1
#filters (N)
…
2x2x1
filter N
2x2xN



### [Page 46]
46/32
Convolution Layer
Convolutional neural network (CNN)
32
3
32
depth
width
height
32x32x3 image -> preserve spatial structure
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Overview



### [Page 47]
47/32
Convolution Layer
Convolutional neural network (CNN)
32
3
32
depth
width
height
5x5x3 filter
32x32x3 image -> preserve spatial structure
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Overview



### [Page 48]
48/32
Convolution Layer
Convolutional neural network (CNN)
32
3
32
depth
width
height
5x5x3 filter
32x32x3 image
Filters always extend the full  
depth of the input volume
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5

Overview



### [Page 49]
49/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32x32x3 image  
5x5x3 filter
32

Overview
Wx+b



### [Page 50]
50/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32x32x3 image  
5x5x3 filter
Wx+b ReLU(Wx+b) 
One number! 
32

Overview



### [Page 51]
51/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
One number = ReLU(Wx+b)

Overview



### [Page 52]
52/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
One number = ReLU(Wx+b)

Overview



### [Page 53]
53/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
One number = ReLU(Wx+b)

Overview



### [Page 54]
54/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
One number = ReLU(Wx+b)

Overview



### [Page 55]
55/32
Convolution Layer
Convolutional neural network (CNN)
3
32
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
convolve (slide) over all  
spatial locations
activation map
1
28
28

Overview



### [Page 56]
56/32
Convolution Layer
Convolutional neural network (CNN)

Overview
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
32
3
32x32x3 image  
5x5x3 filter
convolve (slide) over all  
spatial locations
activation maps
1
28
28
•
Consider a second, green filter



### [Page 57]
57/32
Convolution Layer
Convolutional neural network (CNN)

Overview
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
32
32
3
Convolution Layer
activation maps
6
28
28
•
For example, if we had 6 5x5x3 filters, we`ll get 6 separate activation maps



### [Page 58]
58/32
Convolution Layer
Convolutional neural network (CNN)

Overview
April 18, 2017
Fei-Fei Li & Justin Johnson & Serena Yeung
출처: cs231n_2017_lecture5
•
ConvNet is a sequence of Convolutional Layers, interspersed with activation functions
32
32
3
CONV,  
ReLU
e.g.
6
5x5x3
filters
28
28
6
CONV,  
ReLU
e.g. 10  
5x5x6  
filters
CONV,  
ReLU
….
10
24
24



### [Page 59]
59/32
Convolution



### [Page 60]
60/32
Example of graphical CNN representation
CNN trains the multiple filters (Kernel).



### [Page 61]
61/32



### [Page 62]
62/32
Convolutional Neural Network (CNN) 이론

CNN을이용한classification model 설계시주의사항
62
MLP model 예시
Output node의값이숫자예측확률을나타냄
10x1
784x1



### [Page 63]
63/32
Convolutional Neural Network (CNN) 이론

CNN을이용한classification model 설계시주의사항
63
CNN model 예시
28x28x1
Conv. 1
[5x5x1]x32
24x24x32
Conv. 2
[5x5x32]x32
20x20x32
출력의형태가cube 이므로
예측확률을확인할수없음
…



### [Page 64]
64/32
Convolutional Neural Network (CNN) 이론

CNN을이용한classification model 설계시주의사항
•
일반적으로CNN의feature map을평탄화한이후fully connected layer에입력함
64
LeNet-5 구조
Convolution 1
Convolution 2
Pooling
Pooling
평탄화& FC1
FC2
FC3
Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition



### [Page 65]
65/32
LeNet-5 구조
65
C1
Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition
S2
S4
C5
F6



### [Page 66]
66/32
Applications
66
Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition



### [Page 67]
67
Applications
* Reference: IEEE Trans. PAMI “Image Super-Resolution Using Deep Convolutional Networks” Chao Dong et.al, 2015



### [Page 68]
68
Applications



### [Page 69]
69
Applications



### [Page 70]
70
Applications



### [Page 71]
71
Conclusions
71
Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition



### [Page 72]
72
Artificial Neural Network (ANN) Deep Neural Network (DNN)
Convolution Neural Network (CNN)



### [Page 73]
73/32
References
The world’s top 5 conferences
Ref) https://research.com/conference-rankings/computer-science?p=Q3-2021
(CVPR)
(NIPS)
(ICCV)
(ECCV)
(AAAI)



### [Page 74]
74/32
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Dept. of Computer Engineering
Dong-A University, Busan, Rep. of Korea

