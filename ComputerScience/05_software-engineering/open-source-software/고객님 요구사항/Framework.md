---
aliases: []
course: open-source-software
created: '2024-12-09'
date: '2024-12-09'
semester: 2-2
source: ''
status: seedling
tags:
- cs/open-source
- type/lecture
title: Framework
type: lecture
updated: '2026-05-05'
---


domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
up:: [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM)|문서 객체 모델(DOM)]]
prerequisites:: [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]]
related:: [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/1. 음성 인식 요구 사항|1. 음성 인식 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/배경사진 요구사항|배경사진 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/slot 요구사항|slot 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/TTS 요구 사항|TTS 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항|음성 인식 고객 추가 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/이벤트 이해하기|이벤트 이해하기]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/html, javascript 기초|html, javascript 기초]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기|자바스크립트 객체 다루기]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/연습 문제|연습 문제]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작|웹 시스템 제작]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션|쿠키와 세션]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초 실습2|HTML 기초 실습2]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/연습문제|연습문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/web-programming/2. Spring Boot 개발 환경 세팅/확인문제|확인문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/확인문제|확인문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]]

---
#### Vanilla JS?  jQuery? React? Vue.js?
: 새롭게 시작하는 Front-end 개발 프로젝트에서 위 4가지 중 어느 쪽을 선택할 것인지 검토

- 현재 FE 개발자들의 가장 큰 관심사 중의 하나 
- 한 번 결정되면 되돌리기 어려움. 신중히 접근해야 
- 나무위키 등을 검색해도 내용이 잘 정리되어 있습니다.
---
#### Vanila JS, JQuery 단순 비교
```html
<body>
<h3>바닐라 JS 경우</h3>
	<ul id="list"> 
		<li>Item 1</li>
		<li>Item 2</li>
		<li>Item 3</li> 
	</ul> 
<script> 
	document.getElementById('list').style.background = 'lightgray';
	document.querySelectorAll('li').forEach(i => i.style.color = 'red');
	/* .forEach((i) => {i.style.color ='red'}); 와 동일*/
</script>
</body>
```
```html
<head>
<meta charset="UTF-8"><title>jQuery</title> 
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script></head>
<body>
<h3>jQuery를 사용했을 때</h3> 
	<ul id="list"> 
		<li>Item 1</li> 
		<li>Item 2</li>
		<li>Item 3</li>
	</ul> 
<script>
	$('#list').css('background', 'skyblue');
	$('li').css('color', 'red'); 
</script> 
</body>
```
