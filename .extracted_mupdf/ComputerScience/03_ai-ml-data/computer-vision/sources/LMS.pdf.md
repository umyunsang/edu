## --- [Page 1] ---
컴퓨터비전
Computer Vision
- 최근깊이추정및3D 복원관련기술들-

동아대학교소프트웨어대학AI학과

2026년1학기

임한신

## --- [Page 2] ---
• Depth Anything (CVPR 2024)

• 단안깊이추정(Monocular Depth Estimation) 분야의대표적인파운데이션모델(Foundation Model)
• 준지도학습(Semi-Supervised Learning), Knowledge distillation(Student-Teacher model)을기반으로
학습수행
• Dinov2를encoder로사용
• DPT (Dense Prediction Transformer)를decoder로사용

Depth Anything v3


|  | 최근 깊이 추정 기술 |  |
| --- | --- | --- |


## --- [Page 3] ---
• Depth Anything (CVPR 2024)

• 현재v3 까지나옴(ICLR 2026)
• Metric scale 지원(v1)
• DA3-Giant, DA3-Large, DA3-Base, DA3-Small 의다양한모델사이즈를지원(v2)
• Dual-DPT head로깊이맵과픽셀별광선을동시에예측(v3)

Depth Anything v3


|  | 최근 깊이 추정 기술 |  |
| --- | --- | --- |


## --- [Page 4] ---
• Depth Anything (CVPR 2024)

Depth Anything v3 결과(단안, metric)
Depth Anything v3 결과(1, 2, 13장)


|  | 최근 깊이 추정 기술 |  |
| --- | --- | --- |


## --- [Page 5] ---
• Vision Transformer(비전트랜스포머, ViT)

• 자연어처리에사용되었던Transformer 구조를영상처리에적용한딥러닝모델
• 입력영상을정사각형패치로쪼개고이를1D 벡터로펼친뒤위치벡터를더해주고
transformer에입력
• Transformer encoder에서는패치들간의연관성을계산하는셀프어텐션(Self-Attention) 연산
을수행

Vision Transformer의기본구조


|  | Vision Transformer(ViT) 모델 |  |
| --- | --- | --- |


## --- [Page 6] ---
• DINO(Distillation with No Labels)

• Meta AI(FAIR)에서개발한자기주도학습(Self-Supervised Learning) 기반의비전트랜스포머
(ViT) 학습알고리즘및이를통해학습된ViT encoder
• 동일한구조의두신경망인teacher model과student model을두고두model이같은이미지의
서로다른크롭(글로벌뷰, 로컬뷰)을보고동일한표현을갖도록학습함으로써정답레이
블없이도객체, 경계, 의미론적구조를스스로학습하도록함


|  | Vision Transformer(ViT) 모델 |  |
| --- | --- | --- |


## --- [Page 7] ---
• DINO(Distillation with No Labels)

• 레이블이없어도부분과전체의비주얼을매칭하는과정에서모델내부의셀프어텐션헤드
(Self-Attention Heads)가물체의edge와segmentation mask를자연스럽게발달시키게됨
• 현재v3까지나옴(2025)
• v3에서는약17억장의비지도데이터셋을학습에사용. 또한7B의파라미터모델까지지원
• 영상처리및컴퓨터비전의다양한분야의backbone으로사용되고있음


|  | Vision Transformer(ViT) 모델 |  |
| --- | --- | --- |


## --- [Page 8] ---
• 기본적인목표는3D Scene을Implicit하게구성하는MLP(Multi-Layer Perceptron)을이용하여가상시
점의영상을구성하는것

• 장면을각지점(x, y, z) 및방향(θ,ϕ)을입력하여광도(Radiance)와밀도를출력하는연속적인5D 함수
로표현하며이를카메라광선(Ray)을따라샘플링하여색상(광도)과밀도(불투명도)를예측하도록
학습

• 이후MLP를통해예측된카메라광선상의색상과밀도로부터Volume Rendering 기법을통해가상시
점영상을생성

• 학습시에는실제영상과생성된영상사이의오차를미분해파라미터의학습을수행함


|  | NeRF(Neural Radiance Fields) |  |
| --- | --- | --- |


## --- [Page 9] ---
• Volume Rendering
• 광선을따라3D 공간의밀도와색상을적분하여최종픽셀색을계산하는렌더링기법
• 실제계산에는적분을그대로계산할수없어광선을정해진개수(N)의샘플로쪼개어계산
하는이산적볼륨렌더링(Discrete Volume Rendering) 공식을사용

: 샘플i와i+1 사이의간격

: 샘플구간i에서의불투명도(Opacity). 밀도(σi)가높거나구간(δi)이길수록불투명해짐

: 현재샘플i에도달할때까지앞선구간들을통과하고살아남은투과도.

앞선구간들의투명도((1-α))를모두곱한값


|  | NeRF(Neural Radiance Fields) |  |
| --- | --- | --- |


## --- [Page 10] ---
• NeRF의장점
• 기존의메쉬(Mesh)나포인트클라우드(Point Cloud) 방식으로는표현하기힘들었던투명한
유리, 반사되는금속재질, 털, 연기, 미세한불빛등을잘표현
• 3D 기하학적구조(Topology)의제약이없음

• NeRF의단점
• 장면하나를학습하기위한시간이필요
• 픽셀하나를그릴때마다수많은광선상의점들을신경망에통과시켜야하므로렌더링에
비교적시간이걸림
• 편집및재사용이어려움


|  | NeRF(Neural Radiance Fields) |  |
| --- | --- | --- |


## --- [Page 11] ---
• 3D Gaussian들을Primitive로하여장면을Explicit하게표현

• 각각의3D Gaussian들은일반적으로위치, 회전, 크기, 색상, 불투명도를파라미터로가지게됨

• 3D Gaussian들을Primitive로사용함으로써Differentiable Pipeline이가능하도록설계함

• 렌더링시에는타겟영상의타일별로3D Gaussian들을깊이에따라재배열하였고, 이에기반하
여병렬처리를통한효율적인랜더링이가능


|  | 3D Gaussian Splatting |  |
| --- | --- | --- |


## --- [Page 12] ---
• 3D Gaussian들을Primitive로하여장면을Explicit하게표현

• 각각의3D Gaussian들은일반적으로위치, 회전, 크기, 색상, 불투명도를파라미터로가지게됨

• 3D Gaussian들을Primitive로사용함으로써Differentiable Pipeline이가능하도록설계함

• 렌더링시에는타겟영상의타일별로3D Gaussian들을깊이에따라재배열하였고, 이에기반하
여병렬처리를통한효율적인랜더링이가능

3DGS의기본개념


|  | 3D Gaussian Splatting |  |
| --- | --- | --- |


## --- [Page 13] ---
• 4DGS(4D Gaussian Splatting)
• 3DGS 기술을동적영상으로확장한기술
• 정적장면의표현을기본으로하는3DGS을바탕으로시간의변화에따른gaussian들의변화
를정확하고효율적으로표현하는것을목표로함
• Gaussian들의파라미터들의시간의변화에따른deformation 값을추정하고이를시간에따
라업데이트하거나시간이라는매개변수t를통한함수의값으로표현

4DGS 데모


|  | 3D Gaussian Splatting |  |
| --- | --- | --- |


## --- [Page 14] ---
• Neural Rendering(뉴럴렌더링)
• 컴퓨터그래픽스와인공지능이결합된하이브리드기술로, 딥러닝(인공신경망)을활용하여
3D 공간이나물체를표현하고이를2D 이미지로생성(렌더링)하는기술
• 대표적인기술
• NeRF(Neural Radiance Fields)
• 3D Gaussian Splatting (3DGS)
• 약점
• 장면의표현을위한네트워크또는3D Gaussian들의파라미터들은장면별로최적화또
는학습과정이필요함


|  | Generalizable Neural Rendering |  |
| --- | --- | --- |


## --- [Page 15] ---
• Generalizable Neural Rendering(일반화가능뉴럴렌더링)
• 기존neural rendering 기술의가장큰단점이었던"장면마다새로학습(Per-scene Optimization)
해야한다"는문제를해결하기위한기술
• 사전에대규모데이터로학습된하나의피드-포워드(Feed-forward) 네트워크를사용
• 이를통해별도의추가학습없이실시간또는고속으로3D 공간의정보를즉시추론해냄


|  | Generalizable Neural Rendering |  |
| --- | --- | --- |


## --- [Page 16] ---
• 최근Generalizable Neural Rendering 방법예
• GPS-Gaussian (CVPR 2024)

• 학습데이터를이용하여depth 정보추정네트워크와3D Gaussian들의파라미터를추정하는
네트워크를사전학습시킴
• 추론시에는사전학습된네트워크를통해Feed-forward 방식으로Gaussian 파라미터를추정
및영상합성을수행


|  | Generalizable Neural Rendering |  |
| --- | --- | --- |


## --- [Page 17] ---
• 최근Generalizable Neural Rendering 방법예
• MonoSplat (CVPR 2025)

• 단안깊이추정파운데이션모델의강력한기하학적사전지식(Visual Prior)을활용
• 단안특징들의부족한기하학적일관성을경량어텐션메커니즘(Lightweight Attention)을사
용하여보완


|  | Generalizable Neural Rendering |  |
| --- | --- | --- |


## --- [Page 18] ---
• 최근ViT encoder를backbone으로하는다양한foundation model들이연구되고있음

• 영상으로부터깊이를추정하고3D 구조를추정하는foundation model에대한연구가활발히진
행되고있고이전보다일반화능력및정확도에서상당히좋은결과를보여주고있음

• NeRF, 3DGS 등neural rendering 기술에관한많은연구들이진행되고있고다양한응용분야로
확대되고있음

• 또한최근에는Feed-Forward 방식의Generalizable Neural Rendering 모델들이제안되면서기존의
장면별최적화기반방법의한계를극복하려는연구가활발히이루어지고있음

• NeRF, 3DGS 등neural rendering 뿐만아니라다양한컴퓨터비전기술들이컴퓨터그래픽스분
야와융합되고경계가허물어지고있음


|  | 정리 |  |
| --- | --- | --- |


## --- [Page 19] ---
1.
다음중3D 장면의explicit 표현법이아닌것은?
①메쉬
②3D 가우시안
③포인트클라우드
④네트워크파라미터

2.
다음depth anything에대한설명중잘못된것은?
①Teacher-student model로학습한다.
②DINO를encoder로사용한다.
③카메라파라미터를알아야한다.
④Metric scale을지원한다.


|  | Quiz |  |
| --- | --- | --- |
