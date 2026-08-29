## --- [Page 1] ---

## --- [Page 2] ---

## --- [Page 3] ---

## --- [Page 4] ---

## --- [Page 5] ---

## --- [Page 6] ---

## --- [Page 7] ---

## --- [Page 8] ---
다루기힘듬

## --- [Page 9] ---

## --- [Page 10] ---

## --- [Page 11] ---

## --- [Page 12] ---
인과관계

## --- [Page 13] ---

## --- [Page 14] ---

## --- [Page 15] ---

## --- [Page 16] ---

## --- [Page 17] ---

## --- [Page 18] ---

## --- [Page 19] ---
Chap 4. Chain Rule (aka Product rule)
P(EF) = P(F) P(E|F)

## --- [Page 20] ---
𝐹௘௩ൌ0

𝑈ൌ0

𝑇ൌ0

𝐹௟௨ൌ1

## --- [Page 21] ---
𝐹௟௨ൌ1
𝑈ൌ1

𝑇ൌ1

## --- [Page 22] ---

## --- [Page 23] ---
정확한확률밀도함수를알기힘들거나확률밀도함수가주어졌을때, 해당함수로부
터sample을추출하기어려운경우sampling 하는기본적인방법

Rejection sampling - Wikipedia
https://en.wikipedia.org/wiki/Rejection_sampling

https://untitledtblog.tistory.com/134

## --- [Page 24] ---
Rejection sampling의기본적인동작

• 쉽게샘플을생성할수있는q에서샘플들을생성한뒤에이샘플
들의분포가p를따르도록수정하는것

• 이를통해실제로는q에서샘플이생성되었지만, 그결과는p에
서생성된것처럼만드는것이다. 이때쉽게샘플을생성할수있
도록임의로설정한q를제안분포(proposal distribution)이라
고한다.

• 제안분포는uniform distribution, normal distribution 등이
이용될수있으며, 가능하면p와비슷한형태의확률분포를사용
하는것이좋다.

https://untitledtblog.tistory.com/134

확률분포p와q,M의관계

1) Proposal Distribution

제안분포를설정한다음에는상수M을모든x에대해
p(x)≤Mq(x)이되도록설정한다.

2) Sampling 과정

확률분포q로부터샘플생성및Rejection /

Acceptance 반복수용

A 영역: Rejection
B 영역: Acceptance

## --- [Page 25] ---

## --- [Page 26] ---

## --- [Page 27] ---

## --- [Page 28] ---

## --- [Page 29] ---

## --- [Page 30] ---
(1) samples=sample_a_ton()

## --- [Page 31] ---
(1) samples=sample_a_ton()

## --- [Page 32] ---

## --- [Page 33] ---
(1) samples=sample_a_ton()

## --- [Page 34] ---
(1) samples=sample_a_ton()

## --- [Page 35] ---
(1) samples=sample_a_ton()

## --- [Page 36] ---
(1) samples=sample_a_ton()

## --- [Page 37] ---
(1) samples=sample_a_ton()

## --- [Page 38] ---
(2) samples_observation

## --- [Page 39] ---

## --- [Page 40] ---

## --- [Page 41] ---

## --- [Page 42] ---

## --- [Page 43] ---

## --- [Page 44] ---

## --- [Page 45] ---