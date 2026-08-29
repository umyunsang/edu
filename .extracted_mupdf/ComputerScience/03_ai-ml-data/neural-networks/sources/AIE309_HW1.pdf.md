## --- [Page 1] ---
모든 문제는 풀이 과정을 단계별로 명시하세요. (단, log 는 자연로그를 뜻합니다. )

1. (sigmoid 함수의 backpropagation)

sigmoid 층으로 data batch 묶음 

logloglog
logloglog 가 입력되고, 












일 때 


 를 구하세요.

2. (relu 함수의 backpropagation)

ReLU 층으로 data batch 묶음 




 가 입력되고, 






 일 때




 를 구하세요.

3. (softmaxwithloss 의 backpropagation)

softmaxwithloss 층으로 data batch 묶음 log
log 가 입력되고, target label 이


 일 때 


 를 구하세요.

## --- [Page 2] ---
4. (affine layer 의 backpropagation) 
















, 


,  으로 주어져

있고, 

























일 때, Affine 층의 계산 그래프를 이용하여 


, 


, 

 을 구

하세요.

5. (대칭변환) X=(x1,x2,x3) 가 Y=(x2,x3,x1) 으로 변환되는 layer가 있습니다. 




일 때 


를 구하세요. (힌트: ×

















)

6. (Momentum algorithm) 이변수 함수  를 momentum 방법으로최적화

하고자 합니다. 초기 위치 에서 출발하여 3 step 진행할 때, x1, x2, x3 를 구하세

요. (단 learning rate , momentum 계수 )

7. (NAG algorithm) 이변수 함수  를 NAG 방법으로 최적화하고자 합니다.

초기 위치 에서 출발하여 3 step 진행할 때, x1, x2, x3 를 구하세요. (단 learning

rate , momentum 계수 )

8. (AdaGrad algorithm) 이변수 함수  를 AdaGrad 방법으로 최적화하고자

합니다. 초기 위치 에서 출발하여 2 step 진행할 때, x1, x2를 구하세요. (단

learning rate 

)

9. (RMSProp algorithm) 이변수 함수  를 RMSProp 방법으로 최적화하고

자 합니다. 초기 위치 에서 출발하여 2 step 진행할 때, x1, x2를 구하세요. (단

learning rate 

, forgetting factor 

)

10. (Adam algorithm) 이변수 함수  를 Adam 방법으로 최적화하고자 합니

다. 초기 위치 에서 출발하여 2 step 진행할 때, x1, x2를 구하세요. (단 learning

rate , 

)