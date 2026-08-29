## --- [Page 1] ---
6장

데이터타입


|  |  |  |
| --- | --- | --- |
|  | ISBN 0-321-49362-1 |  |

## --- [Page 2] ---
주제

• 서론
• 기본데이터타입
• 문자스트링타입
• 열거타입
• 배열타입
• 연관배열 
• 레코드타입
• 튜플타입

2

• 리스트타입
• 공용체타입
• 포인터타입과참조타입
• 선택적타입
• 타입검사
• 강타입
• 타입동등
• 이론과데이터타입

## --- [Page 3] ---
서론

• 데이터타입이란?
– 값들의모임과이러한값들에대한미리정의된연산들의

집합으로정의된다

• 모든프로그래밍언어는데이터타입을제공
– 언어에서제공하는데이터타입이실제문제를얼마나잘

표현할수있나?

• 데이터타입의용도
– 오류감지: 타입검사
– 프로그램모듈화지원: 프로그램을구성하는단위로.

클래스또는패키지
– 문서화

3

## --- [Page 4] ---
기본데이터타입

• 대부분의프로그래밍언어는기본데이터
타입(primitive data types)들의집합을제공

• 기본데이터타입이란?
– 다른데이터타입을이용해서정의되지않는타입
– 하드웨어의반영

• 기본데이터타입종류
– 수치타입: 정수, 부동소수점, 십진수, 복소수
– 불리안타입
– 문자타입
4

## --- [Page 5] ---
정수

• 하드웨어의정확한반영
• 다양한크기의정수지원
– Ex. Java의 부호정수: byte, short, int, long

5

## --- [Page 6] ---
부동소수점

• 실수를근사값으로모델링
– ∏, e 와같은무한소수-> 유한한메모리로표현할수없음
– 일부유한소수는유한개의이진수로표현불가능: 0.1

• 소수점이하부분과지수부분으로표현
– IEEE 부동-소수점표준754 형식
– 2가지실수타입: float/ double

6

단정도

배정도

## --- [Page 7] ---
복소수

• 복소수는부동소수점수의쌍(실수부, 허수부) 
으로표현:

– Ex. a+bi, i = root(-1)
– 리터럴 형식: (7 + 3j) in Python
– 복소수산술연산지원

• Ex. Fortran, Python, C99

7

## --- [Page 8] ---
십진수

• 십진수숫자를위한이진수코드를이용하여
문자열스트링과유사하게저장, BCD(binary 
coded decimal) 라고도함.

– 십진수한자리수를표현하기위해4비트필요
– 소수3자리를표현하기위해12비트필요
– 장점: 정확성(십진수값을정확하게표현)
– 단점: 제한된범위, 메모리낭비

• Ex. COBOL, C#, Basic

8

## --- [Page 9] ---
불리안

• 값들의범위는단지 참, 거짓의2가지
– 흔히바이트로구현
– 판독성향상

• 대부분범용언어에서지원
– C99, C++, Java, C#, VB,  Python
– 예외: C89 (0은거짓, 0이아니면참)

9

## --- [Page 10] ---
문자

• 문자데이터는수치코딩으로저장

• 코딩기법: 
– ASCII (8bits): American Standard Code for

Information Interchange)
– 16-bit Unicode (USC-2)

• 1991년Unicode 컨소시엄에서발표
• 세계자연 언어문자대부분포함
• Java에서처음으로도입
• C#, JavaScript, Python 지원
– 32-bit Unicode (USC-4, UTF-32)

10

## --- [Page 11] ---
문자스트링타입

• 문자 스트링타입(character string type)은 
값이일련의문자들로구성

• 설계고려사항
– 기본타입인가? 아니면문자배열인가?
– 스트링의길이가정적인가?  아니면동적인가?

11

## --- [Page 12] ---
문자스트링타입연산

• 전형적인연산들
– 배정
– 비교(=, >, 등)
– 접합
– 부분스트링참조(substring reference)
– 패턴매칭(pattern matching)

12

## --- [Page 13] ---
언어예

• C, C++
– 기본타입이아니고, char 배열로제공

• 스트링은null 문자, ‘\0’ 로끝남
– 스트링연산을표준라이브러리string.h 로제공

• strcpy, strcat, strcmp, strlen
– 안전성?

– C++ 은string 클래스제공

13

char src[] = "Hello World!";
char dest[5];

strcpy(dest, src);

strncpy(dest, src, sizeof(dest)-1);
dest[4] = '\0'; // 널문자처리

std::string s = "Hello";
s += " world";

std::string original = "Hello, world!";
std::string copy = original;

## --- [Page 14] ---
언어예

• Java

– String 클래스: 불변

– StringBuffer 클래스

14

StringBuffer sb = new StringBuffer("Hello");
sb.append(" World"); // 문자열이수정됨(새문자열객체가아님)

String s = "Hello";
s = s + " World"; // 새로운문자열객체생성됨!

## --- [Page 15] ---
언어예

• Python

– 기본타입스트링지원, 
– Java의String 클래스처럼값은불변
– 다양한 스트링연산제공(탐색, 대체, 부분스트링참조,

접합등)

• Perl, JavaScript, PHP

– 정규식기반패턴매칭연산제공(C++, Java, C#,

Python에서 클래스 라이브러리로지원)
 
/[A-Za-z][A-Za-z\d]+/
 
/\d+\.?\d*|\.\d+/

15

## --- [Page 16] ---
스트링길이선택사항

• 정적길이 스트링(static length string)
– 스트링생성시그길이가설정되고고정
– Python, Java

• 제한된 동적길이스트링(limited dynamic length 
string)

– 스트링 선언시고정된최대길이까지의가변적인길이를갖는

것을허용
– C, C++의C 스타일스트링

• 동적길이(dynamic length string)
– 최대길이제한없이가변길이를갖는것을허용
– 최대의 유연성, 동적할당/회수부담
– Perl, JavaScript, C++

16

## --- [Page 17] ---
스트링타입평가

• 작성력향상
– 스트링이문자배열로지원되고, strcpy를위한

함수가제공되지않은경우고려

• 기본타입으로스트링(정적길이) 제공필요
– 동적길이스트링은유연하지만비용부담

• 단순패턴매칭이나접합과같은연산은필수적

17

## --- [Page 18] ---
열거타입

• 열거타입(enumeration type)은 모든가능한
값들이그정의에서제공되는 타입

– 값은 열거상수(enumeration constants)라 불리는

이름상수로표현

• In C,
 
enum days {Mon, Tue, Wed, Thu, Fri, Sat, Sun};

•
C, C++, Java, C#, Python3.4 에서 지원되나,
최근스크립트언어인Perl, JavaScript에서는
지원하지않음

18

## --- [Page 19] ---
예제

19

// in C
enum colors {red, blue, green, yellow, black}; 
 
 
 
 
// 디폴트 내부값은0, 1, …
 
 
 
 
// 정수문맥에서int로강제변환
int main() {
  enum colors myColor = blue, yourColor = red;
  …
  myColor = yourColor + 1; // 적법한가?
  …
  myColor = 4; 
// 적법한가?

}

## --- [Page 20] ---
평가

• 판독성향상: 이름상수가코딩된값보다쉽게
인식

• 신뢰성향상
– 열거타입에대한산술연산이의미가있는가? 
– 열거 타입변수에범위를벗어난값을할당가능한가?

• C, C++, Java, C#의열거타입비교

20