# Index

## lecture

* [01. 프로그래밍 언어론 — 명령어·문법·바인딩·자료형](./01.%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D%20%EC%96%B8%EC%96%B4%EB%A1%A0%20%E2%80%94%20%EB%AA%85%EB%A0%B9%EC%96%B4%C2%B7%EB%AC%B8%EB%B2%95%C2%B7%EB%B0%94%EC%9D%B8%EB%94%A9%C2%B7%EC%9E%90%EB%A3%8C%ED%98%95.md) - 프로그래밍 언어의 목적·발전·구문론에서 이름·바인딩·영역과 자료형까지의 흐름을 정리한다.
* [01. 프로그래밍 언어의 평가 기준과 구현 구조](./01.%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D%20%EC%96%B8%EC%96%B4%EC%9D%98%20%ED%8F%89%EA%B0%80%20%EA%B8%B0%EC%A4%80%EA%B3%BC%20%EA%B5%AC%ED%98%84%20%EA%B5%AC%EC%A1%B0.md) - 프로그래밍 언어의 4대 평가 기준(가독성, 작성력, 신뢰성, 비용), 번역기 구현 모델(컴파일, 인터프리터, 하이브리드 JIT)을 인터랙티브 아키텍처 비교기와 함께 학습한다.
* [02. 프로그래밍 언어 발전사와 패러다임](./02.%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D%20%EC%96%B8%EC%96%B4%20%EB%B0%9C%EC%A0%84%EC%82%AC%EC%99%80%20%ED%8C%A8%EB%9F%AC%EB%8B%A4%EC%9E%84.md) - 명령형, 함수형, 논리형, 객체지향형 패러다임의 역사적 진화와 주요 언어(Fortran, Lisp, C, Java, Python, Rust)의 설계를 학습한다.
* [03. 구문론과 형식 문법(BNF·EBNF)](./03.%20%EA%B5%AC%EB%AC%B8%EB%A1%A0%EA%B3%BC%20%ED%98%95%EC%8B%9D%20%EB%AC%B8%EB%B2%95%28BNF%C2%B7EBNF%29.md) - 문맥 자유 문법(CFG), 배커스-나우르 표기법(BNF) 및 확장 BNF(EBNF), 파스 트리(Parse Tree)와 모호성(Ambiguity) 해소 원리를 학습한다.
* [04. 파싱 기법과 파서 아키텍처](./04.%20%ED%8C%8C%EC%8B%B1%20%EA%B8%B0%EB%B2%95%EA%B3%BC%20%ED%8C%8C%EC%84%9C%20%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98.md) - 하향식 파서(Top-down: 재귀 하강 파서, LL)와 상향식 파서(Bottom-up: 시프트-리듀스, LR)의 원리와 파싱 테이블 구축을 학습한다.
* [05. 이름, 바인딩, 영역과 수명](./05.%20%EC%9D%B4%EB%A6%84%2C%20%EB%B0%94%EC%9D%B8%EB%94%A9%2C%20%EC%98%81%EC%97%AD%EA%B3%BC%20%EC%88%98%EB%AA%85.md) - 변수의 6대 속성(이름, 주소, 타입, 값, 영역, 수명), 정적 vs 동적 바인딩, 정적 스코프 vs 동적 스코프(Lexical vs Dynamic Scope)를 인터랙티브 스코프 분석기로 학습한다.
* [06. 데이터 타입과 타입 시스템](./06.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%83%80%EC%9E%85%EA%B3%BC%20%ED%83%80%EC%9E%85%20%EC%8B%9C%EC%8A%A4%ED%85%9C.md) - 원시 타입, 복합 타입(배열, 레코드, 포인터), 강타입(Strong Typing)과 약타입, 정적/동적 타입 시스템 및 타입 동치성(구조적 vs 이름 기반)을 학습한다.
* [07. 수식과 배정문, 제어 흐름](./07.%20%EC%88%98%EC%8B%9D%EA%B3%BC%20%EB%B0%B0%EC%A0%95%EB%AC%B8%2C%20%EC%A0%9C%EC%96%B4%20%ED%9D%90%EB%A6%84.md) - 연산자 우선순위와 결합성, 참조 투명성(Referential Transparency)과 부작용(Side Effects), 단축 평가(Short-circuit Evaluation) 및 제어문을 학습한다.
* [08. 서브프로그램과 매개변수 전달 기법](./08.%20%EC%84%9C%EB%B8%8C%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%EA%B3%BC%20%EB%A7%A4%EA%B0%9C%EB%B3%80%EC%88%98%20%EC%A0%84%EB%8B%AC%20%EA%B8%B0%EB%B2%95.md) - 서브프로그램의 활성화 레코드(Activation Record), 호출 스택(Call Stack), 매개변수 전달 모델(값 전달, 참조 전달, 이름 전달)의 동작 원리를 학습한다.
* [programming-languages 강의 흐름 지도](./00.%20programming-languages%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 20개의 순서·쪽수·학습 점검을 연결한다.
