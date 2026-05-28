---
aliases: []
course: web-programming
created: '2024-04-09'
date: '2024-04-09'
semester: 2-1
source: ''
status: seedling
tags:
- cs/se
- skill/javascript
- type/lecture
title: 'Spring Boot 기초 확인문제'
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/웹 프로그래밍 인터페이스|웹 프로그래밍 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작|웹 시스템 제작]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초 실습2|HTML 기초 실습2]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초2 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션|쿠키와 세션]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/2. Spring Boot 개발 환경 세팅/Spring Boot 개발 환경 세팅 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/이벤트 이해하기|이벤트 이해하기]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM)|문서 객체 모델(DOM)]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/배경사진 요구사항|배경사진 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/TTS 요구 사항|TTS 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/html, javascript 기초|html, javascript 기초]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/1. 음성 인식 요구 사항|1. 음성 인식 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/Framework|Framework]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/slot 요구사항|slot 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기|자바스크립트 객체 다루기]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/HTML JavaScript 기초 연습문제|연습문제]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항|음성 인식 고객 추가 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기 연습문제|연습문제]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/연습 문제|연습 문제]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/웹 프로그래밍 지식그래프|웹 프로그래밍]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/웹 프로그래밍 지식그래프|웹 프로그래밍]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/웹 프로그래밍 근거 인덱스|웹 프로그래밍 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/doctype html|doctype html]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/웹 프로그래밍|웹 프로그래밍]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/input type|input type]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/background color|background color]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/utf 8|utf 8]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---

### Quiz #1:
클라이언트로부터 서버로 자료 전송 방식 2가지는 ( )방식과 ( )방식입니다.

- 정답:
**클라이언트로부터 서버로 자료를 전송하는 방식은 주로 GET 방식과 POST 방식이 사용됩니다.**

---

### Quiz #2:
주소창에 "/airplane/login"이면 login.html을 띄워주는 Controller의 메소드를 작성하세요.

```java
@GetMapping("/airplane/login")
public String showLoginPage() {
    return "login.html";
}
```

---
### Quiz #3:
wise.html (보내는 쪽), wiseAnswer.html (받는 쪽), controller에 들어갈 메소드 코딩

- ## MyController

```java
@Controller
public class MyController {
	@GetMapping("/wise")
	public String wise() {
	return "wise";
	}
	
	@PostMapping("/wise/answer")
	public String wiseAnswer(@RequestParam("pname") String pname, @RequestParam("word") String word, Model mo) {
	
	mo.addAttribute("pname", pname);
	mo.addAttribute("word", word);
	return "wiseAnswer";
	}
}
```

- ## wise.html

```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body style=background-color:yellow>
<form method="post" action="/wise/answer">
위인 : <input type="text" name="pname"><p>
좋아하는 색 : <select name="word">
<option> 한낱 빛 따위가 어둠의 깊이를 어찌 알겠는가
</select ><p>
<input type ="submit" value="입려">
</form >
</body></html>
```

- ## wiseAnswer.html

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head><meta charset="UTF-8">
<title></title></head>
<body style=background-color:aqua>
<h2> <strong>오늘의 명언</strong></h2><p>
<table border = "1">
	<tr> <th>위인 <td th:text="${pname}">
	<tr> <th>명언 <td th:text="${word}">
</table>
</body>
</html>
```

---

### Quiz #4:
수업자료 3장 ex03Answer.html에서 

(1) 전체바탕색 말고 color 글자 부분의 바탕색만 바꿔 보세요.

```html
<head><title th:text="|${mname}'s color|"></title></head>
<body>
<strong th:text="${mname}">mname</strong>님이 좋아하는 색은 <br>
<strong th:text="${color}" th:style="|background-color: ${color}|">color</strong>입니다.
</body>
```

(2) color글자 색만 바꿔보세요

```html
<head><title th:text="|${mname}'s color|"></title></head>
<body>
<strong th:text="${mname}">mname</strong>님이 좋아하는 색은 <br>
<strong th:text="${color}" th:style="|color: ${color}|">color</strong>입니다.
</body>
```

---

### Quiz #5 
노란 화면에서 입력한 빵 정보가 핑크 화면 에 뜨도록 코딩하세요.

- ### bread.html

```html
<body style="background-color: yellow;">
<form action="/bread/answer" method="get">
    빵종류: <input type="text" name="bread"><p>
    1개당 가격: <input type="number" name="money"><p>
    구입 개수: <select name="count">
        <option>1
        <option>2
        <option>3
        <option>4
        <option>5
    </select> 개 <p>
    <input type="submit" value="구매">
</form>
</body>
```

- ### breadAnswer.html

```html
<body style="background-color: pink;">
    고객님께서 구매하신 빵은<br>
    <strong th:text="${bread}">bread</strong>이며<br>
    <strong th:text="${count}">count</strong>개를 구매하셨으므로<br>
    총 가격은 <strong th:text="${sum}">sum</strong>원 입니다.
</body>
```

---

### Quiz #6

- ### MyController

```java
@Controller
public class MyController {
   
    @GetMapping("/q06a")
    public String q06a() {
        return "q06a";
    }
    
    @GetMapping("/q06")
    public String q06() {
        return "q06";
    }
    
    @GetMapping("/q06aa")
    public String q06aa(@RequestParam("frist") String frist,
                        @RequestParam("second") String second, Model mo) {
        mo.addAttribute("frist", frist);
        mo.addAttribute("second", second);
        return "q06aa";
    }
    
    @GetMapping("/q06b")
    public String q06b() {
        return "q06b";
    }
    
    @GetMapping("/q06bb")
    public String q06bb(@RequestParam("job") String job, Model mo) {
        mo.addAttribute("job", job);
        return "q06bb";
    }
}
```

- ### q06.html

```html
<title>원하는 작품</title>
</head>
<body>
    <h2>선택하세요</h2>
    <p>
        1. <a href="q06a">대기업</a> <br>
        2. <a href="q06b">공무원</a>
    </p>
</body>
```

- ### q06a.html

```html
<title>I기업 선택</title>
</head>
<body style="background-color:aqua">
    <h2>원하는 기업 두 곳 입력</h2>
    <p>
        <form action="/q06aa" method="get">
            1순위:<input type="text" name="frist"><br>
            2순위:<input type="text" name="second"><br>
            <input type="submit" value="확인">
        </form>
    </p>
</body>
```

- ### q06aa.html

```html
<head><title th:text="|${frist} & ${second}|">Ins</title></head>
<style> strong {color : blue} </style>
<body>
    반갑습니다. 조만간 회사에서 만나요! - 
    <strong th:text="${frist}">frist</strong> 인사 팀장 - <p>
    아니오! 우리 회사로 꼭 오세요!! - 
    <strong th:text="${second}">second</strong> 인사 팀장 -
</body>
```

- ### q06b.html

```html
<head> <title>분야 선택</title> </head>
<body style="background-color:yellow">
    <h2>원하는 분야 선택</h2>
    <form action="/q06bb" method="get">
        <select name="job">
            <option>경찰공무원</option>
            <option>소방공무원</option>
            <option>교육공무원</option>
            <option>구청,동사무소</option>
        </select>
        <input type="submit" value="선택">
    </form>
</body>
```

- ### q06bb.html

```html
<head> <title>환영합니다</title> </head>
<body>
    축하합니다!<p>
    <strong th:text="${job}">job</strong> 으로 임용되셨습니다!!
</body>
```
