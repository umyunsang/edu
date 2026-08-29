## --- [Page 1] ---
1/45

Dong-A Univ. (ISPL)

컴퓨터AI공학부

2025년 1학기 머신러닝


|  | Transformer |  |
| --- | --- | --- |


## --- [Page 2] ---
2/45

Background of Language Model

▪정의: 언어라는 현상을 모델링하고자 단어 시퀀스(문장)에 확률을 할당(assign)하는 모델

▪목적: 가장 자연스러운 단어 시퀀스를 찾아내는 것

▪방법: 통계(확률)를 이용한 방법과인공 신경망을 이용한 방법으로 구분

## --- [Page 3] ---
3/45

Background of Language Model

▪기본 동작방식: 이전 단어들이 주어졌을 때 다음 단어를 예측

예시 1:

나는 학교에 
?

갔다
간다
도착했다

예시 2:

인공지능은 데이터를
?

분석한다
학습한다
좋아한다

## --- [Page 4] ---
4/45

Background of Language Model

▪언어 모델의 유형

1. 자기 회귀 언어 모델

- 이전 단어로부터 다음 단어 예측

예시

나는
밥을
?

예측 방향

2. 양방향 언어 모델

예시

나는
?

예측 방향

- 최근 딥러닝 기반 모델에서 사용

- 양쪽 문맥을 활용해 가운데 단어 예측

먹었다

## --- [Page 5] ---
5/45

Background of Language Model

▪언어 모델링: 주어진 단어로부터 모르는 단어를 예측하는 작업

텍스트 데이터 학습

1

패턴 인식

2

다음 단어 예측

3

확률 기반 출력

4

## --- [Page 6] ---
6/45

Background of Language Model

▪언어 모델링: 주어진 단어로부터 모르는 단어를 예측하는 작업

텍스트 데이터 학습

1

패턴 인식

2

다음 단어 예측

3

확률 기반 출력

4

“나는 오늘 학교에 갔다”

“친구는 학교에 간다”

“그는 매일 학교에 간다”

“친구들은 학교에 도착했다”

<학습 데이터>

“학교” 다음에 나오는 단어

“갔다” (1회)

“간다” (2회)

“도착했다” (1회)

<패턴 인식>

“나는 오늘 학교에”

다음에 나오는 단어

<입력>

“갔다”, “간다”, “도착했다”

<확률 할당>

최종출력 ➔“간다”

“갔다”: 25%

“간다”: 50%

“도착했다”: 25%

## --- [Page 7] ---
7/45

Background of Language Model

▪언어 모델의 응용

안녕하세요, 오늘 날씨가 정말 □

1. 텍스트 자동완성

오늘 날씨 어때요?

오늘은 맑고 따뜻합니다!

2. 챗봇 및 대화 시스템

옛날 옛적에…
한 작은 마을에 살던 소년은…
모험을 떠나기로 결심했습니다…

3. 콘텐츠 생성

Hello world
안녕 세상

4. 기계 번역

## --- [Page 8] ---
8/45

Background of Language Model

▪History of Language Model

•
Transformer: 최신 고성능 모델들은 해당 아키텍처 기반으로 설계

RNN
(1986)

Seq2Seq
(NIPS 2014)

Attention
(ICLR 2015)

Transformer
(NIPS 2017)

GPT-1
(2018)

BERT
(NAACL 2019)

GPT-3
(2020)

*BERT: Bidirectional Encoder Representations from Transformers
*GPT: Generative Pre-trained Transformer

## --- [Page 9] ---
9/45

Transformer

## --- [Page 10] ---
10/45

Transformer

인코더
(Encoders)

디코더
(Encoders)

Transformer Model

je suis étudiant

I am a student

▪전체 구조 기계번역 (영어→프랑스어)

•
입력: 영어 문장

•
출력: 프랑스어 문장

## --- [Page 11] ---
11/45

Transformer

인코더
(Encoders)

디코더
(Encoders)

je suis étudiant

I am a student

Encoders

Decoders

▪전체 구조 기계번역 (영어→프랑스어)

•
입력: 영어 문장

•
출력: 프랑스어 문장

Transformer Model

## --- [Page 12] ---
12/45

Transformer

▪전체 구조 기계번역 (영어→프랑스어)

•
학습 시 입력: 영어, 프랑스어 문장

•
학습 시 출력: 프랑스어 문장

Encoders
Decoders

I
am
a
student
<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

Start of Sequence

End of Sequence

1.
영어 문장을 입력

2.
문맥, 의미를 포함하는 고차원 벡터 추출

1.
프랑스어 문장을 입력

2.
인코더 출력을 참고

3.
다음 단어를 예측

4.
최종 프랑스어 문장을 출력

## --- [Page 13] ---
13/45

Transformer

▪Embedding

Encoders
Decoders

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Embedding
Embedding
Embedding
Embedding

<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

1.
딥러닝 모델은 텍스트(단어) 이해 x

2.
임베딩 과정을 통해 숫자(벡터)로 변환

3.
Ex. Student →[0.17,−0.12,0.55,...,0.03]

Start of Sequence

End of Sequence

## --- [Page 14] ---
14/45

Transformer

▪Positional Encoding

Encoders
Decoders

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Embedding
Embedding
Embedding
Embedding

<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

1.
각 단어의 위치 정보를 얻기 위해 사용

2.
positional encoding을 통해 위치 정보 추가

Positional Encoding
Positional Encoding

## --- [Page 15] ---
15/45

Transformer

▪Positional Encoding

Encoders

Embedding
Embedding
Embedding
Embedding

I
am
a
student

1.
각 단어의 위치 정보를 얻기 위해 사용

2.
positional encoding을 통해 위치 정보 추가

Positional Encoding

"I love you" vs "You love I"
Ex.

➔단어는 동일하지만, 순서에 따라 의미가 완전히 달라짐

➔["I”, “love”, “you“] →3개의 벡터가 동시에 처리 →위치x

➔순서 정보가 없으면 트랜스포머는 두 문장을 같다고 인식

Positional

encoding

Embedding

vector

I
am
a
student

+
+
+
+

+

I
am

a
student

➔위치 정보 추가를 위해 덧셈 연산 진행

➔임베딩 벡터 위치, 차원 인식 가능

## --- [Page 16] ---
16/45

Transformer

▪Positional Encoding
"I love you" vs "You love I"
Ex.

➔단어는 동일하지만, 순서에 따라 의미가 완전히 달라짐

➔["I”, “love”, “you“] →3개의 벡터가 동시에 처리 →위치x

➔순서 정보가 없으면 트랜스포머는 두 문장을 같다고 인식

Positional

encoding

Embedding

vector

I
am
a
student

+
+
+
+

+

I
am

a
student

➔위치 정보 추가를 위해 덧셈 연산 진행

➔임베딩 벡터 위치, 차원 인식 가능

✓
𝑝𝑜𝑠: 위치 인덱스 (ex. 0, 1, 2, …)
✓
𝑖: 벡터 차원 인덱스 (ex. 0부터 시작)
✓
𝑑𝑚𝑜𝑑𝑒𝑙: 전체 임베딩 차원 수 (ex. 4)

Embedding vector의 정보 기반으로 
Positional encoding 정보 생성


|  | ( , ) 𝑝𝑜𝑠 𝑖 |  |
| --- | --- | --- |


## --- [Page 17] ---
17/45

Transformer

▪Encoders

Encoders
Decoders

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Embedding
Embedding
Embedding
Embedding

<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

Positional Encoding
Positional Encoding

## --- [Page 18] ---
18/45

Transformer

▪Encoders

•
자연어를 벡터로 변환한 뒤 단어간 유사도 (Relevance) 표현

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding

Encoders

## --- [Page 19] ---
19/45

Transformer

▪Encoders

•
자연어를 벡터로 변환한 뒤 단어간 유사도 (Relevance) 표현

•
각각의 단어별 연관성이 높은 정도를 학습 ➔단어별 Attention을 통해 문맥에 대한 정보 학습

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding

Encoders

## --- [Page 20] ---
20/45

Transformer

▪Encoders

•
자연어를 벡터로 변환한 뒤 단어간 유사도 (Relevance) 표현

•
각각의 단어별 연관성이 높은 정도를 학습 ➔단어별 Attention을 통해 문맥에 대한 정보 학습

Multi-head
Self-Attention
Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding
Add & Norm

Feed Forward

Network

Add & Norm

Encoders

## --- [Page 21] ---
21/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

Multi-head Self-Attention
Scaled Dot-Product Attention

Multi-head
Self-Attention

Add & Norm

Feed Forward

Network

Add & Norm

✓
Query (𝑄): 비교 기준

✓
Key (𝐾): 비교 대상

✓
Value (𝑉): 참조 정보

## --- [Page 22] ---
22/45

Transformer

Query

Keys

Attention

weights

▪Encoder

•
Scaled Dot-Product Attention

✓Q : Query →분석 대상이 되는 단어에 대한 가중치 Vector

✓K : Key →Attention을 수행할 단어 집합

✓V : Value →Attention으로 도출된 weights를 적용한 단어

The animal didn’t cross the street because it was too tired.

그 동물은 길을 건너지 않았다.

기계의 경우 “it”이 “street”인지 “animal”인지 모름

→입력문장내의 단어들끼리 유사도를 계산하여 가중치 부여

❖예시) 영어 →한국어 (Self-attention의 목적 기준)

## --- [Page 23] ---
23/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

student

× 𝑊𝑄(4 × 2)

× 𝑊𝐾(4 × 2)

× 𝑊𝑉(4 × 2)

Multi-head Self-Attention

Weight matrix


|  |  |
| --- | --- |
|  |  |

|  |  |
| --- | --- |


|  |  |
| --- | --- |


## --- [Page 24] ---
24/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

SoftMax

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 25] ---
25/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

SoftMax

𝑄𝑠𝑡𝑢𝑑𝑒𝑛𝑡

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | + 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  | + |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 26] ---
26/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

SoftMax

학습 안정화를 위해 차원 수로 값을 나눔

𝑄𝑠𝑡𝑢𝑑𝑒𝑛𝑡

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 27] ---
27/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

SoftMax

학습에 의미 없는 값들을 0으로 처리

𝑄𝑠𝑡𝑢𝑑𝑒𝑛𝑡

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 28] ---
28/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

SoftMax

차원 수가 깊어 질수록 미분 값이 작아지는 현상을 방지,

확률 값 출력

𝑄𝑠𝑡𝑢𝑑𝑒𝑛𝑡

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 29] ---
29/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

𝑄𝑠𝑡𝑢𝑑𝑒𝑛𝑡

Scaled Dot-Product Attention

× 𝐾𝐼

𝑇

× 𝐾𝑎𝑚
𝑇

× 𝐾𝑎𝑇

𝑇

/ 𝑑𝑘

/ 𝑑𝑘

/ 𝑑𝑘

SoftMax

다른 단어들의 관계에 따라 값 조정


|  |  | 𝑉 𝐼 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | + 𝑉 𝑎𝑚 |  |  |  | + |
| 0. | 1× |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  | + |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑉 𝑎 |  |  |  |  |  |
| 0. | 1× |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

|  |  |
| --- | --- |


|  | / 𝑑 𝑘 |
| --- | --- |
|  |  |

|  |  | 𝑉 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 0. | 4× |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 30] ---
30/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Scaled Dot-Product Attention

## --- [Page 31] ---
31/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

× 𝑊𝑄(4 × 2)

× 𝑊𝐾(4 × 2)

× 𝑊𝑉(4 × 2)

𝑄

Multi-head Self-Attention

I
am
a
student

I
am
a
student

I
am
a
student

I
am

a
student

Weight matrix


|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |

|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |

## --- [Page 32] ---
32/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

×

𝑉

I
am
a
student

a
student
(

(

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑉𝑎𝑙𝑢𝑒


|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑇 × 𝐾 |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

## --- [Page 33] ---
33/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

I
am
a
student

a
student
(

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑉𝑎𝑙𝑢𝑒

Multi-head Self-Attention


|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑇 × 𝐾 |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |
| --- | --- | --- |
| × |  |  |
|  |  |  |
|  |  |  |

| 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟏 | 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟐 |
| --- | --- |


|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |

## --- [Page 34] ---
34/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Multi-head Self-Attention

✓
𝑑𝑚𝑜𝑑𝑒𝑙: 전체 임베딩 차원 수
✓
𝑑𝐴𝑉: 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑉𝑎𝑙𝑢𝑒차원 수
✓
head(h): 2로 가정

I
am
a
student

a
student
(

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑉𝑎𝑙𝑢𝑒


|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑇 × 𝐾 |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |
| --- | --- | --- |
| × |  |  |
|  |  |  |
|  |  |  |

| 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟏 | 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟐 |
| --- | --- |


|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |

## --- [Page 35] ---
35/45

Transformer

▪Encoders

•
Multi-head Self-Attention

•
Scaled Dot-Product Attention

▪Ex.) I am a student

Multi-head Self-Attention

I
am
a
student

a
student
(

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑉𝑎𝑙𝑢𝑒

Weight matrix

×

Multi-head 
Attention matrix


|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | 𝑇 × 𝐾 |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |
| --- | --- | --- |
| × |  |  |
|  |  |  |
|  |  |  |

| 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟏 | 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑉𝑎𝑙𝑢𝑒 𝟐 |
| --- | --- |


|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |

## --- [Page 36] ---
36/45

Transformer

▪Encoders

•
Multi-head Self-Attention

Multi-head
Self-Attention
Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding
Add & Norm

Feed Forward

Network

Add & Norm

Encoders

## --- [Page 37] ---
37/45

Transformer

▪Encoders

•
Residual Connection & Layer Normalization

Multi-head
Self-Attention
Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding
Add & Norm

Feed Forward

Network

Add & Norm

Encoders

## --- [Page 38] ---
38/45

Transformer

▪Encoders

•
Residual Connection & Layer Normalization

Multi-head
Self-Attention

Add & Norm

Feed Forward

Network

Add & Norm

𝑥

F(𝑥)

+

𝐻𝑥= 𝑥+ 𝐹(𝑥)

Residual Connection
Layer Normalization

✓
𝑥𝑖: 입력벡터
✓
𝑘:  벡터 𝑥𝑖의 각 차원
✓
𝜇𝑖: 평균
✓
𝜎𝑖

2: 분산
✓
𝜖: 분모가 0이 되는 것을 방지

## --- [Page 39] ---
39/45

Transformer

▪Encoders

•
Feed Forward Network

Multi-head
Self-Attention
Embedding
Embedding
Embedding
Embedding

I
am
a
student

Positional Encoding
Add & Norm

Feed Forward

Network

Add & Norm

Encoders

## --- [Page 40] ---
40/45

Transformer

▪Encoders

•
Feed Forward Network

Multi-head
Self-Attention

Add & Norm

Feed Forward

Network

Add & Norm

𝑥

𝐹1 = 𝑥𝑊1 + 𝑏1

𝐹2 = max(0, 𝐹1)
Activ. Function: ReLU

𝐹3 = 𝐹2𝑊2 + 𝑏2

## --- [Page 41] ---
41/45

Transformer

▪Decoders

Encoders
Decoders

Embedding
Embedding
Embedding
Embedding

I
am
a
student

Embedding
Embedding
Embedding
Embedding

<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

Positional Encoding
Positional Encoding

## --- [Page 42] ---
42/45

Transformer

▪Decoders

Decoders

Embedding
Embedding
Embedding
Embedding

<sos>
je
suis
étudiant

<eos>
je
suis
étudiant

Positional Encoding

Multi-head
Self-Attention

Add & Norm

Multi-head

Add & Norm

Feed Forward

Network

Add & Norm

𝐾, 𝑉


|  |  |
| --- | --- |
|  | 𝑄 |
|  |  |

## --- [Page 43] ---
43/45

Transformer

## --- [Page 44] ---
44/45

Transformer

▪Transformer 등장 이후 많은 분야에서 연구

## --- [Page 45] ---
45/45

Transformer

▪Transformer 등장 이후 많은 분야에서 연구

## --- [Page 46] ---
46/45

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Division of Computer〮AI Engineering

Dong-A University, Busan, Rep. of Korea