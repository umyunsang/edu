## --- [Page 1] ---
1

http://muntermag.com/2016/09/her-y-el-amor-en-la-era-tecnologica/

농어의무게를예측하라2

선형회귀Linear regression

대표적인회귀알고리즘
비교적간단하고성능이뛰어남
특성이하나인경우어떤직선을학습하는알고리즘

1

2

## --- [Page 2] ---
2

선형회귀

•
모든농어의무게를하나로
예측
•
직선의위치가만약훈련
세트의평균에가깝다면R2는
0에가까운값이됨

•
완전반대로예측함
•
길이가작은농어의무게가
높음
•
길이가큰농어는무게가낮음
•
예측을반대로하면R2 는
음수가됨

•
제일그럴싸함

LinearRegression

y = a × x + b

Model Parameter
-
coef_ :  Coefficient 계수(a 계수)
-
Intercept_ : Weight 가중치(b 절편)

3

4

## --- [Page 3] ---
3

학습한직선그리기

농어의길이15에서50까지직선으로그려봄
길이× 기울기× 절편

데이터결과: 과소적합

다항회귀

2차방정식의그래프를그리기위해길이를제곱한항이훈련세트에추가되어야함

5

6

## --- [Page 4] ---
4

다항회귀

농어의길이를제곱해
원래데이터옆에붙임

모델다시훈련

Model Parameter
-
coef_ :  Coefficient 계수(a 계수)
-
Intercept_ : Weight 가중치(b 절편)

다항식
다항식을이용한선형회귀
다항회귀라고함

7

8

## --- [Page 5] ---
5

학습한직선그리기

그래도과소적합이보임ㅠㅠ

감사합니다

내용출처정보: https://www.hanbit.co.kr/store/books/look.php?p_code=B2002963743

9

10