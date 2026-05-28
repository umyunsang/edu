---
aliases: []
course: open-source-software
created: '2024-11-29'
date: '2024-11-29'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-2
source: ''
status: seedling
tags:
- cs/open-source
- type/lecture
title: slot 요구사항
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/오픈소스 소프트웨어 인터페이스|오픈소스 소프트웨어 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM)|문서 객체 모델(DOM)]]
prerequisites:: [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]]
related:: [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/배경사진 요구사항|배경사진 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/TTS 요구 사항|TTS 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항|음성 인식 고객 추가 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/1. 음성 인식 요구 사항|1. 음성 인식 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/이벤트 이해하기|이벤트 이해하기]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/Framework|Framework]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/html, javascript 기초|html, javascript 기초]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/HTML JavaScript 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기|자바스크립트 객체 다루기]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/연습 문제|연습 문제]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기 연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작|웹 시스템 제작]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초 실습2|HTML 기초 실습2]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션|쿠키와 세션]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초2 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/web-programming/2. Spring Boot 개발 환경 세팅/Spring Boot 개발 환경 세팅 확인문제|확인문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론|3. 구문론]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/오픈소스 소프트웨어 지식그래프|오픈소스 소프트웨어]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/오픈소스 소프트웨어 지식그래프|오픈소스 소프트웨어]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/오픈소스 소프트웨어 근거 인덱스|오픈소스 소프트웨어 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/utf 8|utf 8]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/meta charset|meta charset]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/doctype|doctype]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/link rel|link rel]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/open-source-software/input type|input type]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---
## slot원본.html
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>슬롯머신</title>
  <link rel="stylesheet" href="slot.css">
</head>
<body>
  <div class="slot-machine">
    <span class="slot" id="slot1">🍒</span>
    <span class="slot" id="slot2">🍋</span>
    <span class="slot" id="slot3">🍊</span>
  </div>
  <button onclick="startSlotMachine()">슬롯머신 돌리기</button>
  <h3 id="result">ready</h3>
  <script>
    const symbols = ["🍒", "🍋", "🍊", "⭐", "🔔"];
    const slotarr = document.querySelectorAll(".slot");
    const r = document.getElementById("result");

    function getRandomSymbol() {
      return symbols[Math.floor(Math.random() * symbols.length)];
    }

    function startSlotMachine() {
      const finalSymbols = [];
      for (let i = 0; i < slotarr.length; i++) 
        finalSymbols.push(getRandomSymbol());
      
      let iterationCount = 0;

      function spinSlots() {
        iterationCount++;
        slotarr.forEach((s) => {s.innerHTML = getRandomSymbol();});

        if (iterationCount >= 10) {
          clearInterval(intervalId);
          slotarr.forEach((s, i) => { s.innerHTML = finalSymbols[i];});

          /* 결과 출력 */
          if (finalSymbols[0] == finalSymbols[1] && finalSymbols[1] == finalSymbols[2]) 
            r.textContent = "Congratulation!!";
          else if (finalSymbols[0] == finalSymbols[1] || finalSymbols[1] == finalSymbols[2] || finalSymbols[0] == finalSymbols[2]) 
            r.textContent = "2개 맞았음";
          else 
            r.textContent = "꽝!";
        }
      }

      const intervalId = setInterval(spinSlots, 100); // 0.1초
    }
  </script>
</body>
</html>
```

#### 다른방법도 가능 (시험예상)
```js
const slotarr = document.querySelectorAll('.slot');
/* 1. getElementsByClassName('slot') */
/* 2. getElementsTag__() */

for(let s of slotarr)
	s.innerHTML = getRandomSymbol();

for(let i=0; i<slotarr.lenth; i++)
	slotarr[i].innerHTML = finalSymbols[i];
```
- forEach는 querySelectorAll()로 가져온 배열만 호환 됨
- forof나 for문을 사용해야 함
#### 함수 A 안에 함수 a를 설정할 수 있습니다.
```js
function A(){
	...
	function a(){
		a()함수도 A()안에 있으므로
		A()의 지역변수도 공유할 수 있습니다.
		10번 반복되면 멈추라는 명령문도 여기 있음.
	}
	const intervalId = setInterval(a, 100);
}
```
- 버튼 onclick하면 A()함수가 돌아감 하지만 0.1초마다 반복되는 부분은 a()

#### 요구사항 1
- 현재 10회 랜덤반복 후 결과그림이 고정됩니다.
- 0.1초마다 반복하는 동안 화면에 1부터 10까지 출력되도록 해주세요
#### 요구사항 2
- 현재 startSlotMachinge 함수 안에 spinSlots 함수가 있습니다.
- spingSlots함수를 화살표 함수로 바궈보세요
#### 요구사항 3
- 현재 특수문자 그림 3개를 사용합니다. 그래서 div 안에 span태그 3개를 사용하였습니다.
- 그런데 진짜 그림으로 바구려고 합니다. span 3개 말고 img 3개로 바꿔보세요.
#### 요구사항 4
- 버튼 1회 천원이라고 가정합니다.
- 그리고 배팅금액을 입력할 수 있도록 수정하세요. (숫자인지 먼저 체크하고, 그 다음 최소금액 이상인지 체크해서, 1000원 이상 숫자를 입력할 때까지 반복)
#### 요구사항 5
- 화면 윗부분에 잔액이 보이고, 꽝이면 cost만큼 줄어들고(즉, 천원 마이너스), 2개 맞으면 cost만큼 늘어나고(즉, 천원 플러스), 잭팟이면 cost의 10배만큼 잔액 늘어나도록(즉, 만원 플러스) 변경하세요. 
- 화면에 출력할 때 천단위 쉼표 넣으려면 money가 잔액 변수일 때 ... ${money.toLocaleString()} ... 이렇게 하시면 됩니다.
#### 요구사항 6
- 잔액이 cost보다 적으면 다시 배팅하라는 팝업창이 뜨고, 프로그램 다시 시작 (다시 시작하는 방법은 여러 가지인데 location.reload(); 가 그 중 하나입니다.)
#### 최종 slot
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>슬롯머신</title>
  <link rel="stylesheet" href="slot.css">
  <style>
    #money {border: dashed; font-size: small; padding: 10px 20px; color:gray;}
  </style>
</head>
<body>
  <span id="money"></span> <!-- 요구사항 5 -->
  <div class="slot-machine"> <!-- 요구사항 3 -->
    <img class="slot" id="slot1" src="/3. 문서 객체 모델/static/cat.jpg">
    <img class="slot" id="slot2" src="/3. 문서 객체 모델/static/dog.jpg">
    <img class="slot" id="slot3" src="/3. 문서 객체 모델/static/duck.jpg">
  </div>
  <button onclick="startSlotMachine()">슬롯머신 돌리기</button>
  <h3 id="result">ready</h3>
  <h4 id="icount">&nbsp;</h4>  <!-- 요구사항 1 -->
  <script>
    const cost = 1000;
    let money = 0;
    const symbols = ["cat", "dog", "duck", "koala", "사과"];
    const slotarr = document.querySelectorAll(".slot");
    const r = document.getElementById("result");
  
    while (1){ // 요구사항 4
      money = parseInt(prompt("배팅 금액을 입력하세요 (버튼 1회에 비용 1000원)"));
      if (isNaN(money)) alert("숫자를 입력하세요.");
      else if (money < 1000) alert("1000원 이상을 입력해주세요.");
      else break;
    }
    document.getElementById("money").textContent = `잔액 : ${money.toLocaleString()} 원`;
  
    function getRandomSymbol() {
      return symbols[Math.floor(Math.random() * symbols.length)];
    }
  
    function startSlotMachine() {
      if (money == 0){ // 요구사항 6
        alert("잔액이 부족합니다. 처음부터 다시 시작하세요.");
        location.reload(); 
        return;  
      }
      else if (money < cost){ // 요구사항 6
        alert("잔액이 부족합니다. 처음부터 다시 시작하세요 (잔액 500원은 카운터에서 환불받으실 수 있습니다.)");
        location.reload();
        return;
      }
  
      money -= cost; // 요구사항 5
      document.getElementById("money").textContent = `잔액 : ${money.toLocaleString()} 원`;
  
      const finalSymbols = [];
      for (let i = 0; i < slotarr.length; i++)
        finalSymbols.push(getRandomSymbol());
      let iterationCount = 0;
  // 요구사항 2
      const spinSlots = () => { // 요구사항 1
        iterationCount++;
        document.getElementById("icount").textContent = iterationCount;
        slotarr.forEach((s) => {
           s.src = `/3. 문서 객체 모델/static/${getRandomSymbol()}.jpg`;
        });
  
        if (iterationCount >= 10) {
          clearInterval(intervalId);
          slotarr.forEach((s, i) => {
            s.src = `/3. 문서 객체 모델/static/${finalSymbols[i]}.jpg`;
          });
  
          /* 결과 출력 */
          if (finalSymbols[0] == finalSymbols[1] && finalSymbols[1] == finalSymbols[2]){
            r.textContent = "Congratulation!!";
            money += cost * 10; // 요구사항 5
          }
          else if (finalSymbols[0] == finalSymbols[1] || finalSymbols[1] == finalSymbols[2] || finalSymbols[0] == finalSymbols[2]) {
            r.textContent = "2개 맞았음";
            money += cost; // 요구사항 5
          }  
          else
            r.textContent = "꽝!";
        }
        document.getElementById("money").textContent = `잔액 : ${money.toLocaleString()} 원`; // 요구사항 5
      }
  
      const intervalId = setInterval(spinSlots, 100); // 0.1초
    }
  </script>
</body>
</html>
```
