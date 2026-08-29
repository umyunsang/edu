## --- [Page 1] ---
Time-Series Analysis

Yongjae Lee
Department of Industrial Engineering

## --- [Page 2] ---
Topics

▪1. Introduction to Time-Series Analysis

▪2. Deep Learning Models for Time-Series Analysis

▪3. Case Studies

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
1

## --- [Page 3] ---
Section 1

Introduction to Time-Series Analysis

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
2

## --- [Page 4] ---
시계열 분석

▪시계열 분석: 여러 시점에 관측된 실험 데이터를

분석하는 것

– 기존에 사용되는 대부분의 통계적 기법이 바로 적용되기

어려움

• 기존에는 많은 경우 관측 데이터가 독립적이고 동일한 
분포를 가진다고 가정 (independent and identically 
distributed, i.i.d.)
• 하지만 시계열 데이터는 인접하여 관측된 데이터의 경우 
분명한 관계성이 있음

– 따라서 시계열 분석에서는 이러한 부분을 잘 다룰 수

있어야 함

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
3

## --- [Page 5] ---
가장 오래 된 시계열 차트

▪약 10, 11세기에 그려진 것으로 알려진 천체의 움직임

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
4

## --- [Page 6] ---
예시1: Los Angeles의 연간 강우량

▪간단한 관측

– 매년 강우량 변동이 꽤 크다
– 1880년대에 아주 큰 강우량을 기록한 적이 있다

▪간단한 질문

– 전해와 이듬해의 강우량에 관계성이 있을까?

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
5

## --- [Page 7] ---
예시1: Los Angeles의 연간 강우량

examination
Considerable variation in rainfall amount over

Exceptionally high in 1983

›

▪간단한 질문

– 전해와 이듬해의 강우량에 관계성이 있을까? 아마 아닐 것

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
6

## --- [Page 8] ---
예시2: 연간 토끼 개체 수

▪간단한 관측

– 인접한 관측 값들이 유사한 값을 가짐

▪간단한 질문

– 전해와 이듬해의 개체 수에 관계성이 있을까?

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
7

## --- [Page 9] ---
예시2: 연간 토끼 개체 수

Rough examination
Neighboring values are closely related

▪간단한 질문

– 전해와 이듬해의 개체 수에 관계성이 있을까? 그런 듯

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
8

## --- [Page 10] ---
예시3: 월간 오일 필터 판매량

▪간단한 관측

– 상당히 변동 폭이 큰 듯

▪간단한 질문

– 데이터에 계절성(seasonality)이 있을까?

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
9

## --- [Page 11] ---
예시3: 월간 오일 필터 판매량

Rough examination

Quite fluctuating

▪간단한 질문

– 데이터에 계절성(seasonality)이 있을까? 그런 듯

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
10

## --- [Page 12] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
11

f
x
f(x)

## --- [Page 13] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
12

x
f(x)
ML/AI

## --- [Page 14] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
13

ML/AI
Dog

## --- [Page 15] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
14

ML/AI

## --- [Page 16] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
15

## --- [Page 17] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
16

ML/AI

## --- [Page 18] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
17

ML/AI

???

## --- [Page 19] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
18

ML/AI

???
조금 거칠게 말해서 함수의 성질이 잘 지켜는 데이터
즉, 하나의 입력값에 하나의 출력값이 잘 대응되는 데이터는

정상성(stationarity)이 있는 데이터라고 할 수 있으며,

이러한 데이터는 예측이 잘 되는 편

## --- [Page 20] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
19

x
f(x)
LLM

## --- [Page 21] ---
예측 모형의 원리

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
20

미국 대선은 
누가 이겼니?
도널드 트럼프
LLM

## --- [Page 22] ---
Model-driven 방식

▪전통적 기법들은 대부분 model-driven이라 볼 수 있음

– 계산되는 수식이 큰 틀에서 이미 결정되어있음
– 데이터 분포에 대한 가정이 있는 경우가 많음

– 장점

• 도메인 지식을 활용하기 쉬움
• 분석 결과를 일반화 하기 쉬움

– 단점

• 데이터나 분석 환경이 복잡할 경우엔 사용하기 어려움

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
21

## --- [Page 23] ---
Model-driven 방식

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
22

f
x
f(x)

ARIMA

GARCH

Ex)

## --- [Page 24] ---
Data-driven 방식

▪AI 기법은 대부분data-driven 방식으로 볼 수 있음

– 계산 수식이나 데이터의 확률 분포에 대해 특별한 가정이

없음

– 장점

• 다양한 변수 사이의 복잡한 (일반적으로 비선형적인) 
관계성을 반영할 수 있음

– 단점

• 충분한 양의 데이터가 필요
• 결과를 해석하기가 다소 어려움

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
23

## --- [Page 25] ---
Data-driven 방식

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
24

f
x
f(x)

Ex)

Random Forest
Neural Network

## --- [Page 26] ---
Section 2

Deep Learning Models 
for Time Series Analysis

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
25

## --- [Page 27] ---
Artificial neural networks

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
26

## --- [Page 28] ---
2.1

RNN-based models

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
27

## --- [Page 29] ---
Recurrent neural networks (RNN)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
28

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## --- [Page 30] ---
Recurrent neural networks (RNN)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
29

▪RNNs are widely used in

http://datahacker.rs/003-rnn-architectural-types-of-different-recurrent-neural-networks/

## --- [Page 31] ---
Vanishing gradient of RNNs

▪기본RNN 구조의 한계점

– 같은 구조를 계속해서 반복하기 때문에 input의 영향이

뒤로 갈 수록 점점 미미해지거나 아니면 오히려 너무 
지나치게 증폭 될 수 있음
– 간단하게 말하자면, RNN이 초기 입력값을잊어버림

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
30

## --- [Page 32] ---
Long short term memory (LSTM)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
31

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## --- [Page 33] ---
Neural ODE

▪RNN과 LSTM 모두 관측값이 동일한

간격으로 관츨되었다고 가정

▪NeuralODE (Chen et al., 2018)는 일정하지

않게 관측된 데이터도 다룰 수 있도록 
hidden state를 continuous하게 설정

– ANN은 hidden state의 derivative (미분값,

변화량)을 학습

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
32

## --- [Page 34] ---
RNN-based models

▪Example: DeepAR (Salinas et al., 2020) by Amazon

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
33

https://doi.org/10.1016/j.ijforecast.2019.07.001

## --- [Page 35] ---
RNN-based models

▪Example: DeepAR (Salinas et al., 2020) by Amazon

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
34

https://doi.org/10.1016/j.ijforecast.2019.07.001

## --- [Page 36] ---
2.2

Generative models

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
35

## --- [Page 37] ---
Generative models

▪Generative models vs. discriminative models

– 간단하게 말하자면

• Discriminative models 데이터의 종류를 판별하는 모형

› 예를 들면 사진을 바탕으로 개인지 고양이인지 구분

• Generative models은 새로운 데이터를 만들어내는 모형

› 예를 들어 여러 이미지를 학습 후 새로운 이미지를 만들어냄

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
36

Source: Google Developers

## --- [Page 38] ---
Generative Adversarial Networks (GAN)

▪Goodfellow et al. (2014)가 제안한 두 네트워크가 게임을

하듯이 경쟁하는 생성모형

– Generator (generative network)

• 데이터 분포를 학습하여 새로운 데이터를 만들어내고자 함
• Discriminator를 속이는 것이 목표
– Discriminator (discriminative network)

• Generator가 생성한 가짜 데이터와 진짜 데이터를 구분하는 
것이 목표

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
37

## --- [Page 39] ---
Generative Adversarial Networks (GAN)

▪Examples: StyleGAN (Karras et al., 2019) by NVIDIA

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
38

## --- [Page 40] ---
Generative Adversarial Networks (GAN)

▪Examples: CycleGAN (Zhu et al., 2017)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
39

## --- [Page 41] ---
Generative Adversarial Networks (GAN)

▪Examples: TimeGAN (Yoon et al., 2019)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
40

https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

## --- [Page 42] ---
Generative Adversarial Networks (GAN)

▪Examples: QuantGANs (Wiese et al., 2020)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
41

•https://doi.org/10.1080/14697688.2020.1730426

## --- [Page 43] ---
Generative Adversarial Networks (GAN)

▪Examples: TadGAN (Geiger et al., 2020)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
42

https://ieeexplore.ieee.org/abstract/document/9378139

## --- [Page 44] ---
Diffusion models

▪데이터에 점점 노이즈를 더하면 언젠가는 완전한

노이즈가 됨

▪그 반대 과정을 학습할 수 있다면 노이즈로부터 실제

같은 데이터를 생성해낼 수 있음

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
43

Image source: Ho et al. (2020)

Image source: S. Orbell’s blog

## --- [Page 45] ---
Diffusion models

▪Example: TimeGrad (Rasul et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
44

https://proceedings.mlr.press/v139/rasul21a.html

## --- [Page 46] ---
Diffusion models

▪Example: TimeGrad (Rasul et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
45

https://proceedings.mlr.press/v139/rasul21a.html

## --- [Page 47] ---
2.3

Attention-based models

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
46

## --- [Page 48] ---
Sequence-to-sequence models

▪sequence-to-sequence (seq2seq) models에서는 input이

sequence로 들어가며 hidden state가 update되고, 이를 
바탕으로 output을 sequence로 출력

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
47

Image source: Lil’Log

## --- [Page 49] ---
Sequence-to-sequence models

▪Input이 길어지게 되면 이를 기억하기가 어려움

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
48

Image source: Lil’Log

## --- [Page 50] ---
Attention mechanism

▪따라서 전체 input을 하나로 요약하기보다, input에서

어떤 부분을 더 ‘집중(attend)’해야하는지 판단하여 사용

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
49

Image source: Lil’Log

## --- [Page 51] ---
Attention mechanism

▪따라서 전체 input을 하나로 요약하기보다, input에서

어떤 부분을 더 ‘집중(attend)’해야하는지 판단하여 사용

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
50

Image source: Cheng et al. (2016)

## --- [Page 52] ---
Attention is all you need (Transformers)

▪Vaswani et al. (2017)는 RNN cell 없이

attention만으로 구성된 Transformer 
모형을 제안

– 여러 task에서 매우 높은 성능을 보임

– 가장 큰 장점은 병렬화 시켜 모델을

매우 크게 만들기가 용이하여, 
엄청나게 많은 양의 데이터를 
학습하는 것이 가능해짐

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
51

## --- [Page 53] ---
Attention is all you need (Transformers)

▪Example: Temporal Fusion Transformers (Lim et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
52

https://doi.org/10.1016/j.ijforecast.2021.03.012

## --- [Page 54] ---
Attention is all you need (Transformers)

▪Example: Temporal Fusion Transformers (Lim et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
53

https://doi.org/10.1016/j.ijforecast.2021.03.012

## --- [Page 55] ---
Attention is all you need (Transformers)

▪Example: Temporal Fusion Transformers (Lim et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
54

Fig. E.7. TFT one-step-ahead forecasts for real
ized volatility across major stock indices follo
wing the incidence of COVID-19. From the tar
get values in purple, we note the regime shift 
following the sharp increase in index volatiliti
es in March 2020, which later decay to elevat
ed baseline levels over 2020. While volatilitie
s in March 2020 do appear at or around the 0
.975 quantile forecast, the TFT maintains reas
onable uncertainty estimates over the regime 
shift – with most values lying within the 95% 
credible interval – demonstrating its ability to 
adapt to changing temporal dynamics. (For in
terpretation of the references to colour in thi
s figure legend, the reader is referred to the 
web version of this article.)

https://doi.org/10.1016/j.ijforecast.2021.03.012

## --- [Page 56] ---
Attention is all you need (Transformers)

▪Example: Informer (Zhou et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
55

https://doi.org/10.1609/aaai.v35i12.17325

## --- [Page 57] ---
Attention is all you need (Transformers)

▪Example: Autoformer (Wu et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
56

https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html

## --- [Page 58] ---
Attention is all you need (Transformers)

▪Example: Autoformer (Wu et al., 2021)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
57

https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html

## --- [Page 59] ---
2.4

LLM-based models

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
58

## --- [Page 60] ---
LLM-based models

▪LLMTime[1]

– Tokenization for the numeric values
– (-): 연산/대수관계를 제대로 이해하는 방식이 아니라, 한계가 명확함

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
59

[1] Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems, 36, 19622-19635.

## --- [Page 61] ---
LLM-based models

▪TimeLLM [2]

– Projecting TS to single-modality embedding
– (-): Foundation language model을 backbone으로 사용하기에, modality

gap/ TS -> emb 로 부터 발생하는 정보 손실

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
60

[1] Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., ... & Wen, Q. (2023). Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728.

## --- [Page 62] ---
LLM-based models

▪TimesNet [3]

– TS -> frequency / amplitude로의 변환을 통해 long / short context 정보를

동시에 반영하려 함.
– (-): domain shift 발생 시 모델이 가지고 있는 inductive bias가 상당 부분

사라짐-> adaptation에 어려움을 겪을 가능성 높음.

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
61

[3] Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2022). Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186.

## --- [Page 63] ---
LLM-based models

▪Time-VLM

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
62

Zhong, S., Ruan, W., Jin, M., Li, H., Wen, Q., & Liang, Y. (2025). Time-VLM: Exploring multimodal vision-language models for augmented ti
me series forecasting. arXiv preprint arXiv:2502.04395.

## --- [Page 64] ---
LLM-based models

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
63

Kong, Y., Yang, Y., Hwang, Y., Du, W., Zohren, S., Wang, Z., ... & Wen, Q. (2025). Time-mqa: Time series multi-task question answering wit
h context enhancement. arXiv preprint arXiv:2503.01875.

## --- [Page 65] ---
Section 3

Case Studies

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
64

## --- [Page 66] ---
Motivation

65

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
65

## --- [Page 67] ---
Motivation

66

External context might reduce 
uncertainty and non-stationarity

of financial time-series

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
66

## --- [Page 68] ---
Paper 1

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
67

Lee, J., Jeon, H., Bae, H., & Lee, Y. (2025, November). Return Prediction for Mean-Variance P
ortfolio Selection: How Decision-Focused Learning Shapes Forecasting Models. In Proceedings 
of the 6th ACM International Conference on AI in Finance (pp. 114-122).
https://dl.acm.org/doi/full/10.1145/3768292.3770423

## --- [Page 69] ---
Paper 1

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
68

Lee, J., Jeon, H., Bae, H., & Lee, Y. (2025, November). Return Prediction for Mean-Variance P
ortfolio Selection: How Decision-Focused Learning Shapes Forecasting Models. In Proceedings 
of the 6th ACM International Conference on AI in Finance (pp. 114-122).
https://dl.acm.org/doi/full/10.1145/3768292.3770423

## --- [Page 70] ---
Paper 1

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
69

Lee, J., Jeon, H., Bae, H., & Lee, Y. (2025, November). Return Prediction for Mean-Variance P
ortfolio Selection: How Decision-Focused Learning Shapes Forecasting Models. In Proceedings 
of the 6th ACM International Conference on AI in Finance (pp. 114-122).
https://dl.acm.org/doi/full/10.1145/3768292.3770423

## --- [Page 71] ---


| Paper 1 |  |
| --- | --- |
|  |  |
| Lee, J., Jeon, H., Bae, H., & Lee, Y. (2025, November). Return Prediction for Mean-Variance P UNIST Financial Engineering Lab. ortfolio Selection: How Decision-Focused Learning Shapes Forecasting Models. In Proceedings 70 of the 6th ACM International C(hontftepresn:c/e/ foen lAaI bin. Fuinnainscte. a(pcp.. 1k1r4)-122). https://dl.acm.org/doi/full/10.1145/3768292.3770423 |  |

## --- [Page 72] ---
Paper 2

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
71

Kim, J., Tae, I., & Lee, Y. (2025, November). Estimating Covariance for Global Minimum Varian
ce Portfolio: A Decision-Focused Learning Approach. In Proceedings of the 6th ACM Internation
al Conference on AI in Finance (pp. 105-113).
https://dl.acm.org/doi/full/10.1145/3768292.3770378

## --- [Page 73] ---
Paper 2

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
72

Kim, J., Tae, I., & Lee, Y. (2025, November). Estimating Covariance for Global Minimum Varian
ce Portfolio: A Decision-Focused Learning Approach. In Proceedings of the 6th ACM Internation
al Conference on AI in Finance (pp. 105-113).
https://dl.acm.org/doi/full/10.1145/3768292.3770378

## --- [Page 74] ---
Paper 2

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
73

Kim, J., Tae, I., & Lee, Y. (2025, November). Estimating Covariance for Global Minimum Varian
ce Portfolio: A Decision-Focused Learning Approach. In Proceedings of the 6th ACM Internation
al Conference on AI in Finance (pp. 105-113).
https://dl.acm.org/doi/full/10.1145/3768292.3770378

## --- [Page 75] ---
Paper 2

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
74

Kim, J., Tae, I., & Lee, Y. (2025, November). Estimating Covariance for Global Minimum Varian
ce Portfolio: A Decision-Focused Learning Approach. In Proceedings of the 6th ACM Internation
al Conference on AI in Finance (pp. 105-113).
https://dl.acm.org/doi/full/10.1145/3768292.3770378

## --- [Page 76] ---
Paper 3

▪Applied decision-focused learning to

MVO with LLM-based prediction

75

(working paper)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
75

Hwang, Y., Kong, Y., Zohren, S., & Lee, Y. (2025). Decision-informed neural networks with larg
e language model integration for portfolio optimization. arXiv preprint arXiv:2502.00828.
https://arxiv.org/abs/2502.00828

## --- [Page 77] ---
Paper 3

▪Applied DFL to MVO with LLM-based

prediction

76

(working paper)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
76

Hwang, Y., Kong, Y., Zohren, S., & Lee, Y. (2025). Decision-informed neural networks with larg
e language model integration for portfolio optimization. arXiv preprint arXiv:2502.00828.
https://arxiv.org/abs/2502.00828

## --- [Page 78] ---
Paper 3

▪Applied DFL to MVO with LLM-based

prediction

77

(working paper)

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
77

Hwang, Y., Kong, Y., Zohren, S., & Lee, Y. (2025). Decision-informed neural networks with larg
e language model integration for portfolio optimization. arXiv preprint arXiv:2502.00828.
https://arxiv.org/abs/2502.00828

## --- [Page 79] ---
Paper 4

▪Text – Time-Series pair dataset

▪Pair text and time-series based on

semantic and multi-level analysis

78

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
78

Lee, J., Park, S., Lim, T. Y., Lee, S., Seo, J., Kang, D., ... & Ahn, W. (2026). FinTexTS: Financi
al Text-Paired Time-Series Dataset via Semantic-Based and Multi-Level Pairing. KDD 2026 Dat
asets & Benchmarks Track
https://arxiv.org/abs/2603.02702

## --- [Page 80] ---
Paper 4

▪Text– Time-Series pair dataset

▪Pair text and time-series based on

semantic and multi-level analysis

79

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
79

Lee, J., Park, S., Lim, T. Y., Lee, S., Seo, J., Kang, D., ... & Ahn, W. (2026). FinTexTS: Financi
al Text-Paired Time-Series Dataset via Semantic-Based and Multi-Level Pairing. KDD 2026 Dat
asets & Benchmarks Track
https://arxiv.org/abs/2603.02702

## --- [Page 81] ---
Prediction markets

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
80

## --- [Page 82] ---
Prediction markets

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
81

## --- [Page 83] ---
Prediction markets

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
82

## --- [Page 84] ---
Prediction markets

Prediction markets transform unstructured global and local events

(e.g., Fed rate cuts, vocabulary of keynote)
into tradable and liquid data with a unique skin-in-the-game accuracy

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
83

## --- [Page 85] ---
Paper 5

▪Used LLM to filter and re-rank lead-la

g relationships found by Granger caus
ality to improve generalizability

▪On Kalshi Economics markets, our

hybrid approach consistently 
outperforms the statistical baseline

84

Kim, S., Kim, M., Kwon, J., Kim, Y., Kagan, N., Lee, J. W., Levy, O., Lopez-Lira, A., Lee, Y. & C
hoi, C. (2026). LLM as a Risk Manager: LLM Semantic Filtering for Lead-Lag Trading in Predicti
on Markets. The 64th Annual Meeting of the Association for Computational Linguistics (ACL 202
6), Industry Track, Accepted
https://arxiv.org/abs/2602.07048

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
84

## --- [Page 86] ---
Paper 5

▪Used LLM to filter and re-rank lead-la

g relationships found by Granger caus
ality to improve generalizability

▪On Kalshi Economics markets, our

hybrid approach consistently 
outperforms the statistical baseline

85

Kim, S., Kim, M., Kwon, J., Kim, Y., Kagan, N., Lee, J. W., Levy, O., Lopez-Lira, A., Lee, Y. & C
hoi, C. (2026). LLM as a Risk Manager: LLM Semantic Filtering for Lead-Lag Trading in Predicti
on Markets. The 64th Annual Meeting of the Association for Computational Linguistics (ACL 202
6), Industry Track, Accepted
https://arxiv.org/abs/2602.07048

UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
85

## --- [Page 87] ---
UNIST Financial Engineering Lab.

(https://felab.unist.ac.kr)
86