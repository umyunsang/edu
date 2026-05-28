---
aliases: []
course: open-source-software
created: '2024-11-24'
date: '2024-11-24'
semester: 2-2
source: ''
status: seedling
tags:
- cs/open-source
- type/lecture
title: TTS 요구 사항
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/오픈소스 소프트웨어 인터페이스|오픈소스 소프트웨어 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM)|문서 객체 모델(DOM)]]
prerequisites:: [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]]
related:: [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/배경사진 요구사항|배경사진 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/slot 요구사항|slot 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/1. 음성 인식 요구 사항|1. 음성 인식 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항|음성 인식 고객 추가 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/이벤트 이해하기|이벤트 이해하기]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/Framework|Framework]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/html, javascript 기초|html, javascript 기초]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/HTML JavaScript 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기|자바스크립트 객체 다루기]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/연습 문제|연습 문제]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작|웹 시스템 제작]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초 실습2|HTML 기초 실습2]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션|쿠키와 세션]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초2 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/2. Spring Boot 개발 환경 세팅/Spring Boot 개발 환경 세팅 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/오픈소스 소프트웨어 지식그래프|오픈소스 소프트웨어]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/오픈소스 소프트웨어 지식그래프|오픈소스 소프트웨어]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/오픈소스 소프트웨어 근거 인덱스|오픈소스 소프트웨어 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/utf 8|utf 8]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/meta charset|meta charset]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/doctype|doctype]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/link rel|link rel]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/input type|input type]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---
## TTS원본.html
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TTS원본</title>
</head>
<body>
    <h2>텍스트를 읽어드립니다</h2>
    <textarea id="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
    
    <script>
        function readText() {
            const a = document.getElementById("box");
            let text = a.value.trim();
            if (text === "") {
                alert("읽을 텍스트를 입력하세요.");
                return;
            }
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    </script>
</body>
</html>
```

#### 고객님의 요구사항
- 입력박스를 2개 만들고 디자인은 아래 화면처럼 해주세요. (가운데 정렬, 맨 위 파란 글자, 박스 내 글자크기, 버튼 변경(색, 글자크기, 여백) 
- 한 박스에라도 글자 입력 시에는 읽어주세요. 두 박스 모두 비면 팝업창 띄워주세요. 
- 위 박스부터 차례대로 텍스트 읽어주세요.
![](../../../../image/Pasted%20image%2020241124151009.png)

#### 팀장의 요구사항 
- (1) 박스가 2개 뿐이므로 반복문 사용하지 말고 getElementById() 2번 사용해서 편하게 코딩한 소스 TTS수정후.html 
- (2) 지금은 2개지만 앞으로 확장될 가능성을 대비해서 getElementsByClassName() 사용해서 배열로 받은 후 for-of 반복문으로 코딩한 소스 (TTS수정후for.html)

>[!TTS수정후.html]
```html
<style>
    body {text-align: center;}
    h2 {color: blue;}
    button {background-color: lime;
            font-size: 1em;
            padding: 0px 20px 0px 20px;}
</style>
<body>
    <h2>여러 텍스트를 이어서 읽어드립니다</h2>
    <textarea id="box1" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <textarea id="box2" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
<script>
function readText() {
    const a = document.getElementById("box1");
    const b = document.getElementById("box2");
    let text_a = a.value.trim();
    let text_b = b.value.trim();
    if (text_a == "" && text_b == "") {
        alert("읽을 텍스트를 입력하세요.");
        return;
    }
    const utterance_a = new SpeechSynthesisUtterance(text_a);
    const utterance_b = new SpeechSynthesisUtterance(text_b);
    window.speechSynthesis.speak(utterance_a);
    window.speechSynthesis.speak(utterance_b);
}
</script>
</body>
```

>[!TTS수정후for.html]
```html
<body>
    <h2>여러 텍스트를 이어서 읽어드립니다</h2>
    <textarea class="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <textarea class="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
    
<script>
    function readText() {
    const arr = document.getElementsByClassName("box");
    let main_text = "";

    for (let i = 0; i < arr.length; i++) {
        let text = arr[i].value.trim();
        main_text += text;
    }
    if (main_text !== "") {
        const utterance = new SpeechSynthesisUtterance(main_text);
        window.speechSynthesis.speak(utterance);
    }
    else {
        alert("읽을 텍스트를 입력하세요.");
        return;
    }
    }
</script>
</body>
```
