## --- [Page 1] ---
2024-03-01

1

Variable Selection
(=Feature Selection)

빅데이터분석
컴퓨터AI공학부

천세진


| Variable Selection(변수 선택)  강의 목표: 다른 변수선택기법을 이해하는것이 목표  변수선택기법: 데이터셋 내 제일 좋은 특징(features)를 선택하는방법  데이터 과학에서핵심 과정 중의 하나임 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 2 | 동아대학교 |

## --- [Page 2] ---
2024-03-01

2

Feature vector

3


| Feature vector: Collection of numerical features  dwa |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 4 | 동아대학교 |

## --- [Page 3] ---
2024-03-01

3


| Feature space |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 5 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 6 | 동아대학교 |

## --- [Page 4] ---
2024-03-01

4


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 7 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 8 | 동아대학교 |

## --- [Page 5] ---
2024-03-01

5


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 9 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 10 | 동아대학교 |

## --- [Page 6] ---
2024-03-01

6


| Property of Feature space |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 11 | 동아대학교 |

| Property of Feature space |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 12 | 동아대학교 |

## --- [Page 7] ---
2024-03-01

7

Feature selection

13


| You are the Coach for Soccer team  The best player in each position (Best features)  Don’t want many players play the same position (Multicollinearity)  다중공선성 문제 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 14 | 동아대학교 |

## --- [Page 8] ---
2024-03-01

8


| PySpark 기반 변수선택 기법 주성분분석 카이제곱선택 증거가중치를사용한정보값 특이값분해 모델기반 선택 트랜스포머 투표기반선택 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 15 | 동아대학교 |

| 변수선택 기법의 직관적인 표현 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 16 | 동아대학교 |

## --- [Page 9] ---
2024-03-01

9


| Exploratory Data Analysis(EDA)  탐색적 데이터 분석: 모델링 전에 필수적인활동  데이터 내존재하는특징(Characteristics)과 패턴(patterns)을 식별하기 위해데이터를 분석하는 과정  Cardinality와 Missing Values  데이터가 주어졌을 때, 먼저해야 체크해야 하는부분 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 17 | 동아대학교 |

| Cardinality  변수의 고유한 값에 대한 개수  집값을예측하는 프로그램을 모델링  Home type은유용한 정보인가? Home price dataset |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 18 | 동아대학교 |

## --- [Page 10] ---
2024-03-01

10


| Missing Values(결측값)  정보의 사라진 부분  Missing at random (MAR): 무작위결측  관측된데이터와결측된데이터간관계, 예) Home type and HOA fees  Missing completely at random(MCAR): 완전 무작위결측  관측된데이터와결측된데이터간관계가없음, 예) Price  Missing not at random(MNAR): 비무작위 결측  관측된데이터와결측된데이터간관계가있음 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 19 | 동아대학교 |

| Missing Values  정보의 사라진 부분 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 20 | 동아대학교 |

## --- [Page 11] ---
2024-03-01

11


| MAR의 경우에 대한 처리 방법  결측된 값을 채우는 방법을 사용  데이터 대치법(Imputation) 혹은데이터 보간법(Interpolation)  다음과 같은 형태를 고려해야함  Mean, median, or mode imputation  Model-based imputation  Multiple imputation  비즈니스 로직사용  Drop features with significant missing data (variable selection)  Drop rows with missing values(추천 안됨) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 21 | 동아대학교 |

| MNAR  데이터는수집되지 않았음  예로, built year  데이터 수집과정에서는 존재하지 않는값  새로운특징(features)를찾는과정 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 22 | 동아대학교 |

## --- [Page 12] ---
2024-03-01

12


| Dataset 다운로드  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 23 | 동아대학교 |

| Bank Marketing Dataset  약 4만 5천 건의 데이터셋  고객정보가주어졌을때, 우리 마케팅 프로그램에가입 가능성?  x: 고객정보, y: 마케팅 프로그램가입여부 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 24 | 동아대학교 |

## --- [Page 13] ---
2024-03-01

13


| Bank Marketing Dataset |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 25 | 동아대학교 |

| Bank Marketing Dataset |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 26 | 동아대학교 |

## --- [Page 14] ---
2024-03-01

14


| https://rpubs.com/johnakwei/330635 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 27 | 동아대학교 |

| 변수선택 기법  주어진 고객 데이터에 대한 검증  Cardinality Check  Missing Value Check  String values  Data Scaling |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 28 | 동아대학교 |

## --- [Page 15] ---
2024-03-01

15


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 29 | 동아대학교 |

| Load Dataset |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 30 | 동아대학교 |

## --- [Page 16] ---
2024-03-01

16


| InferSchema  df.printSchema()  df.dtypes Without inferSchema With inferSchema |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 31 | 동아대학교 |

| Descriptive Statistics  df.describe().toPandas() |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 32 | 동아대학교 |

## --- [Page 17] ---
2024-03-01

17


| Count the records by group  df.groupBy('education').count().show() |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 33 | 동아대학교 |

| Count the records by target  df.groupBy(target variable name).count().show() _ _ |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 34 | 동아대학교 |

## --- [Page 18] ---
2024-03-01

18


| Group by Multiple columns  df.groupBy(['education',target variable name]).count().show() _ _ |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 35 | 동아대학교 |

| Group by Multiple columns  from pyspark.sql.functions import *  df.groupBy(target variable name).agg({'balance':'avg', 'age': _ _ 'avg'}).show() |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 36 | 동아대학교 |

## --- [Page 19] ---
2024-03-01

19


| Cardinality Check |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 37 | 동아대학교 |

| Cardinality Check |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 38 | 동아대학교 |

## --- [Page 20] ---
2024-03-01

20


| Missing value check |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 39 | 동아대학교 |

| Techniques for Converting Strings to Numbers.  개별적으로각 컬럼마다진행  OneHotEncoder  StringIndexer  Weighted Index |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 40 | 동아대학교 |

## --- [Page 21] ---
2024-03-01

21


| Converting All String Columns to be Number ones.  One hot encoder  컬럼의 가중치를고려하지 않음 Education EducationPrimary EducationSecondary EducationTertiary EducationUnkown _ _ _ _ primary 1 0 0 0 secondary 0 1 0 0 tertiary 0 0 1 0 Unknown 0 0 0 1 Primary 1 0 0 0 secondary 0 1 0 0 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 41 | 동아대학교 |

| Education | EducationPrimary _ | EducationSecondary _ | EducationTertiary _ | EducationUnkown _ |
| --- | --- | --- | --- | --- |
| primary 1 0 0 0 |  |  |  |  |
| secondary 0 1 0 0 |  |  |  |  |
| tertiary 0 0 1 0 |  |  |  |  |
| Unknown 0 0 0 1 |  |  |  |  |
| Primary 1 0 0 0 |  |  |  |  |
| secondary 0 1 0 0 |  |  |  |  |

| Converting All String Columns to be Number ones.  StringIndexer  순서가 무시되기때문에 사용할때주의 Education EducationIndex _ primary 0 secondary 1 tertiary 2 Unknown 3 Primary 0 secondary 1 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 42 | 동아대학교 |

| Education | EducationIndex _ |
| --- | --- |
| primary 0 |  |
| secondary 1 |  |
| tertiary 2 |  |
| Unknown 3 |  |
| Primary 0 |  |
| secondary 1 |  |

## --- [Page 22] ---
2024-03-01

22


| 변환 방법  #1 String 변수를 가진 특징(컬럼)을선택  #2 특정 변환기를선택된 컬럼에 적용  #3 Assemble Features into Feature Vectors  #4 Apply Scaler to Feature Vectors Assembled |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 43 | 동아대학교 |

| #1 String 변수를 가진 특징(컬럼)을 선택 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 44 | 동아대학교 |

## --- [Page 23] ---
2024-03-01

23


| #2 특정 변환기를 선택된 컬럼에 적용 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 45 | 동아대학교 |

| #2 특정 변환기를 선택된 컬럼에 적용 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 46 | 동아대학교 |

## --- [Page 24] ---
2024-03-01

24


| #3 Assemble Features into Feature Vectors |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 47 | 동아대학교 |

| #3 Assemble Features into Feature Vectors df.describe().toPandas() |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 48 | 동아대학교 |

## --- [Page 25] ---
2024-03-01

25


| #3 Assemble Features into Feature Vectors df.describe().toPandas() |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 49 | 동아대학교 |

| #3 Assemble Features into Feature Vectors |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 50 | 동아대학교 |

## --- [Page 26] ---
2024-03-01

26


| #4 Apply Scaler to Feature Vectors Assembled  StandardScaler |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 51 | 동아대학교 |

| #4 Apply Scaler to Feature Vectors Assembled |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 52 | 동아대학교 |

## --- [Page 27] ---
2024-03-01

27

Built-in Variable Selection Process:

Without Target

54

Built-in Variable Selection Process:

Without Target


| #4 Apply Scaler to Feature Vectors Assembled |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 53 | 동아대학교 |

## --- [Page 28] ---
2024-03-01

28


| Without Target  Principal Component Analysis  Singular Value Decomposition  Model-based Feature Selection |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 55 | 동아대학교 |

| Principal Component Analysis  대규모 데이터셋에서패턴을식별  대부분의 정보를포함하는 변수들을 확인  데이터셋에 대한선형 표현 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 56 | 동아대학교 |

## --- [Page 29] ---
2024-03-01

29


| Example: Sample Data 참가자ID Quality Realiablity 1 10 6 2 9 4 3 8 5 4 3 3 5 2 2 6 1 1 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 57 | 동아대학교 |

| 참가자ID | Quality | Realiablity |
| --- | --- | --- |
| 1 | 10 | 6 |
| 2 | 9 | 4 |
| 3 | 8 | 5 |
| 4 | 3 | 3 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

| Example: Sample Data 참가자ID Quality Realiablity 1 10 6 2 9 4 3 8 5 4 3 3 Mean 5 2 2 6 1 1 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 58 | 동아대학교 |

| 참가자ID | Quality | Realiablity |
| --- | --- | --- |
| 1 | 10 | 6 |
| 2 | 9 | 4 |
| 3 | 8 | 5 |
| 4 | 3 | 3 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

## --- [Page 30] ---
2024-03-01

30


| Example: Sample Data 참가자ID Quality Realiablity 1 10 6 2 9 4 3 8 5 4 3 3 5 2 2 6 1 1 특정 Feature의 분산을 최대한으로 유지하는 선형 함수를 PC1 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 59 | 동아대학교 |

| 참가자ID | Quality | Realiablity |
| --- | --- | --- |
| 1 | 10 | 6 |
| 2 | 9 | 4 |
| 3 | 8 | 5 |
| 4 | 3 | 3 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

| PCA (3/16=1/5) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 60 | 동아대학교 |

## --- [Page 31] ---
2024-03-01

31


| Loading scores for PCs |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 61 | 동아대학교 |

| Loading scores for PCs |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 62 | 동아대학교 |

## --- [Page 32] ---
2024-03-01

32


| PCA with StandardScaler |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 63 | 동아대학교 |

| Singular Value Decomposition(특이값 분해) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 64 | 동아대학교 |

## --- [Page 33] ---
2024-03-01

33


| Singular Value Decomposition(특이값 분해)  데이터 압축 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 65 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 66 | 동아대학교 |

## --- [Page 34] ---
2024-03-01

34


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 67 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 68 | 동아대학교 |

## --- [Page 35] ---
2024-03-01

35


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 69 | 동아대학교 |

| SVD(특이값 분해) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 70 | 동아대학교 |

## --- [Page 36] ---
2024-03-01

36


| SVD(특이값 분해) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 71 | 동아대학교 |

| ChiSq Selector(카이제곱 선택) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 72 | 동아대학교 |

## --- [Page 37] ---
2024-03-01

37


| ChiSq Selector(카이제곱 선택)  χ2 = Σ (관측값 - 기댓값)2 / 기댓값  적합도 검정(Goodness of fit)  Null Hypothesis vs. Alternative Hypothesis X2: 카이제곱, O: 관측된데이터, E: 기대되는데이터 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 73 | 동아대학교 |

| ChiSq Selector(카이제곱 선택)  생존자 비교: 성별과 생존자 간에 관계가 있는가?  여성생존자: 67.9% (231/340)  남성생존자: 32% (109/340)  Null Hypo.: 성별과 생존자간에는관계없다 Titanic example: Gender survived |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 74 | 동아대학교 |

## --- [Page 38] ---
2024-03-01

38


| ChiSq Selector(카이제곱 선택)  기대값 계산 Gender Survived Expected Value Female No (312*549)/889 192.67 Female Yes (312*340)/889 119.32 Male No (577*549)/889 356.32 Male Yes (577*340)/889 220.67 Titanic example: Gender survived |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 75 | 동아대학교 |

| Gender | Survived |  | Expected Value |
| --- | --- | --- | --- |
| Female | No | (312*549)/889 | 192.67 |
| Female | Yes | (312*340)/889 | 119.32 |
| Male | No | (577*549)/889 | 356.32 |
| Male | Yes | (577*340)/889 | 220.67 |

| ChiSq Selector(카이제곱 선택)  카이제곱값 Gender Survived Expected Value Observed Chi-Square Female No 192.67 81 64.72 Female Yes 119.32 231 104.51 Male No 356.32 468 34.99 Male Yes 220.67 109 56.51 260.75 Titanic example: Gender survived |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 76 | 동아대학교 |

| Gender | Survived | Expected Value | Observed | Chi-Square |
| --- | --- | --- | --- | --- |
| Female | No | 192.67 | 81 | 64.72 |
| Female | Yes | 119.32 | 231 | 104.51 |
| Male | No | 356.32 | 468 | 34.99 |
| Male | Yes | 220.67 | 109 | 56.51 |
|  |  |  |  | 260.75 |

## --- [Page 39] ---
2024-03-01

39


| ChiSq Selector(카이제곱 선택)  카이제곱값 Gender Survived Expected Value Observed Chi-Square Female No 192.67 81 64.72 Female Yes 119.32 231 104.51 Male No 356.32 468 34.99 Male Yes 220.67 109 56.51 260.75 Titanic example: Gender survived |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 77 | 동아대학교 |

| Gender | Survived | Expected Value | Observed | Chi-Square |
| --- | --- | --- | --- | --- |
| Female | No | 192.67 | 81 | 64.72 |
| Female | Yes | 119.32 | 231 | 104.51 |
| Male | No | 356.32 | 468 | 34.99 |
| Male | Yes | 220.67 | 109 | 56.51 |
|  |  |  |  | 260.75 |

| P-value(유의 확률)  테스트한결과를 얻을 확률  Null hypothesis testing 참이라고 할 때,  P-value 가0.05 이하 이라면, null hypothesis is (기각)rejected |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 78 | 동아대학교 |

## --- [Page 40] ---
2024-03-01

40


| P-value 선택  0에 가까울수록, 주어진 셸간 의미가있다  df는(rows-1) * (cols -1) https://people.richland.edu/james/lecture/m170/tbl-chi.html |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 79 | 동아대학교 |

| Tardiness Check 귀무가설: 매요일마다동등하게지각인원이발생한다 대안가설: 매요일마다동등하게지각인원이발생하지않는다. Significance level(검정수준) 5% Week of day / Monday Tuesday Wednesday Thursday Friday Observed 9 15 8 13 8 Expected 10 10 10 10 10 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 80 | 동아대학교 |

| Week of day / | Monday | Tuesday | Wednesday | Thursday | Friday |
| --- | --- | --- | --- | --- | --- |
| Observed | 9 | 15 | 8 | 13 | 8 |
| Expected | 10 | 10 | 10 | 10 | 10 |

## --- [Page 41] ---
2024-03-01

41


| ChiSq Selector |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 81 | 동아대학교 |

| ChiSq Selector |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 82 | 동아대학교 |

## --- [Page 42] ---
2024-03-01

42


| Model-based Feature Selection  Make a computer to distinguish between dogs and cats  In Spark, Tree-based methods is provided by default  Decision Tree  Random Forest  Gradient Boosting |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 83 | 동아대학교 |

| Random Forest  Supervised machine learning algorithm |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 84 | 동아대학교 |

## --- [Page 43] ---
2024-03-01

43


| Random Forest |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 85 | 동아대학교 |

| Random Forest |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 86 | 동아대학교 |

## --- [Page 44] ---
2024-03-01

44

Custom-built Variable Selection

88


| PySpark 기반 변수 선택 기법 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 87 | 동아대학교 |

## --- [Page 45] ---
2024-03-01

45


| Contents  Information Value using Weight of Evidence  Custom Transformers  Voting-based Selection |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 89 | 동아대학교 |

| Information Value(IV) using Weight of Evidence(WoE)  Variable transformation과selection에서강력한 도구  신용점수측정에 주로 사용됨  만약고객이 지불불이행(default on a payment)을할지 예측  Events와Non Events로분류  Events: 지불불이행을한고객  Non Events: 지불불이행을하지않은고객 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 90 | 동아대학교 |

## --- [Page 46] ---
2024-03-01

46


| Information Value(IV) using Weight of Evidence(WoE)  Handles Missing values  Handles Outliers  Transformation은분포는 로그값에기반하기 때문에,  로그 기반의 회귀 값들과정렬 가능함  불필요한변수를 요구하지 않음  적절한 bin 테크닉을사용하여, 독립과 종속 변수간에  monotonic 관계를 만들어낼 수 있음 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 91 | 동아대학교 |

| Information Value(IV) using Weight of Evidence(WoE) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 92 | 동아대학교 |

## --- [Page 47] ---
2024-03-01

47


| Information Value(IV) using Weight of Evidence(WoE) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 93 | 동아대학교 |

| Monotonicity |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 94 | 동아대학교 |

## --- [Page 48] ---
2024-03-01

48


| Spearman Correlation  두 데이터 간의 상관관계  방향과 연관성을계산할 수있음  d는두데이터간의 차이 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 95 | 동아대학교 |

| Spearman Correlation =0.64 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 96 | 동아대학교 |

## --- [Page 49] ---
2024-03-01

49


| Monotonicity  Bin을 효과적으로설계하는것이 중요 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 97 | 동아대학교 |

| 멜버른 집값 데이터 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 98 | 동아대학교 |

## --- [Page 50] ---
2024-03-01

50


| 멜버른 집값 데이터  집의 유형(Type)과 관계된 특징을 추출하는 것이 목표 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 99 | 동아대학교 |

| Colab에서 Notebook 생성  Spark setup은 동일하게 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 100 | 동아대학교 |

## --- [Page 51] ---
2024-03-01

51


| 데이터 확인  집의 유형  1: house, villa의주거형태  0: 다른주거형태  데이터 유형  가격  욕실/방 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 101 | 동아대학교 |

| 데이터 확인  문자형, 숫자형 데이터 분리  불필요한데이터필드 제거  다음사항도확인  카디날리티  결측값 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 102 | 동아대학교 |

## --- [Page 52] ---
2024-03-01

52


| 라이브러리 Import  WOE 계산 함수  Pearson correlation을위한 Monotonic 함수  특징별로WOE 계산 실행 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 103 | 동아대학교 |

| WOE 계산 함수 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 104 | 동아대학교 |

## --- [Page 53] ---
2024-03-01

53


| Pearson correlation을 위한 Monotonic 함수 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 105 | 동아대학교 |

| 특징별로 WOE 계산 및 실행 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 106 | 동아대학교 |

## --- [Page 54] ---
2024-03-01

54

Custom Transformer

108


| WOE & IV 수행 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 107 | 동아대학교 |

## --- [Page 55] ---
2024-03-01

55


| Custom Transformer  Pyspark Pipeline Model  다수의 Stage로구성하여결과 도출  OOP 개념을사용하여, 워크플로우/파이프라인을 구축  End-to-end process |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 109 | 동아대학교 |

| 파이프라인의 주요 개념  DataFrame: This ML API uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types.  E.g., a DataFramecould have different columns storing text, feature vectors, true labels, and predictions  Transformer: is an algorithm which can transform one DataFrameinto another DataFrame.  E.g., an ML model is a Transformer which transforms a DataFramewith features into a DataFramewith predictions.  Estimator: is an algorithm which can be fit on a DataFrameto produce a Transformer.  E.g., a learning algorithm is an Estimator which trains on a DataFrameand produces a model.  Pipeline: chains multiple Transformers and Estimators together to specify an ML workflow.  Parameter: All Transformers and Estimators now share a common API for specifying parameters. |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 110 | 동아대학교 |

## --- [Page 56] ---
2024-03-01

56


| 파이프라인의 예제 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 111 | 동아대학교 |

| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 112 | 동아대학교 |

## --- [Page 57] ---
2024-03-01

57


| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 113 | 동아대학교 |

| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 114 | 동아대학교 |

## --- [Page 58] ---
2024-03-01

58


| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 115 | 동아대학교 |

| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 116 | 동아대학교 |

## --- [Page 59] ---
2024-03-01

59


| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 117 | 동아대학교 |

| Custom Correlation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 118 | 동아대학교 |

## --- [Page 60] ---
2024-03-01

60

Voting-based Selection

120


| Pipeline for custom transformers |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 119 | 동아대학교 |

## --- [Page 61] ---
2024-03-01

61


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 121 | 동아대학교 |

| Overview |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 122 | 동아대학교 |

## --- [Page 62] ---
2024-03-01

62


| Results |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 123 | 동아대학교 |

| Methods |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 124 | 동아대학교 |

## --- [Page 63] ---
2024-03-01

63


| Voting-based selection |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 125 | 동아대학교 |

| Summary  처음 데이터셋을받았을 때,  초기전처리과정 (결측값, 오류확인등)  변수선택을 신중하게 해야함  타겟값이주어질때, 주어지지않을때로분리하여생각할것  비슷한 데이터셋을계속 받는다면,  파이프라인을 설계해보자 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 126 | 동아대학교 |

## --- [Page 64] ---
2024-03-01

64

Singular Value Decomposition

Appendix #1

127


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 128 | 동아대학교 |

## --- [Page 65] ---
2024-03-01

65


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 129 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 130 | 동아대학교 |

## --- [Page 66] ---
2024-03-01

66


|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 131 | 동아대학교 |

|  |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 132 | 동아대학교 |

## --- [Page 67] ---
2024-03-01

67

Information Value and

Weight of Evidence

Appendix #2


| References  Easiest Way to Understanding Singular Value Decomposition (SVD) with Python: numpy.linalg.svd  https://www.youtube.com/watch?v=5d67893GJao&t=1339s |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 133 | 동아대학교 |

## --- [Page 68] ---
2024-03-01

68


| Weight of Evidence(WoE)  Feature의 예측력을 이해 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 135 | 동아대학교 |

| Information Value  해당 Feature의 예측력을하나의 값으로 표현 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 136 | 동아대학교 |

## --- [Page 69] ---
2024-03-01

69

Correlation/Coefficient

Appendix #3


| Weight of Evidence(WoE)  WoE for a specific class Age group Good(1) Bad(0) %Good %Bad WOE %G - %B IV 20-30 2000 400 30-40 3500 201 40-50 4200 300 60-70 1200 96 Total |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 137 | 동아대학교 |

| Age group | Good(1) | Bad(0) | %Good | %Bad | WOE | %G - %B | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20-30 | 2000 | 400 |  |  |  |  |  |
| 30-40 | 3500 | 201 |  |  |  |  |  |
| 40-50 | 4200 | 300 |  |  |  |  |  |
| 60-70 | 1200 | 96 |  |  |  |  |  |
| Total |  |  |  |  |  |  |  |

## --- [Page 70] ---
2024-03-01

70


| Correlation  measures the strength of association between two variables and the direction of the relationship  Pearson, Spearman, Kendal  Also called “coefficient” |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 139 | 동아대학교 |

| Non-parametric testing 1 2 3 4 SCORE Y 5 1 2 3 4 5 ABSENCE Y |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 140 | 동아대학교 |

## --- [Page 71] ---
2024-03-01

71


| Parametric testing  Correlation vs p-value SCORE Y ABSENCE Y |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 141 | 동아대학교 |