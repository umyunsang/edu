# Index

## lecture

* [01. 알고리즘 설계 — 효율성·분할정복·동적계획법·탐욕·NP](./01.%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%20%EC%84%A4%EA%B3%84%20%E2%80%94%20%ED%9A%A8%EC%9C%A8%EC%84%B1%C2%B7%EB%B6%84%ED%95%A0%EC%A0%95%EB%B3%B5%C2%B7%EB%8F%99%EC%A0%81%EA%B3%84%ED%9A%8D%EB%B2%95%C2%B7%ED%83%90%EC%9A%95%C2%B7NP.md) - 알고리즘 개요와 효율성 분석에서 브루트포스·분할정복·동적계획법·탐욕·백트래킹·NP까지의 흐름을 정리한다.
* [01. 알고리즘의 개요와 문제 해결 프레임워크](./01.%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9D%98%20%EA%B0%9C%EC%9A%94%EC%99%80%20%EB%AC%B8%EC%A0%9C%20%ED%95%B4%EA%B2%B0%20%ED%94%84%EB%A0%88%EC%9E%84%EC%9B%8C%ED%81%AC.md) - 알고리즘의 정의, 5대 조건, 문제 해결 설계-분석 생명주기 및 유클리드 호제법 최대공약수 알고리즘을 인터랙티브 시뮬레이터와 함께 학습한다.
* [02. 알고리즘 효율성 분석과 점근적 표기법](./02.%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%20%ED%9A%A8%EC%9C%A8%EC%84%B1%20%EB%B6%84%EC%84%9D%EA%B3%BC%20%EC%A0%90%EA%B7%BC%EC%A0%81%20%ED%91%9C%EA%B8%B0%EB%B2%95.md) - 시간 복잡도와 공간 복잡도의 기본 원리, 점근적 표기법(빅오, 빅오메가, 빅세타), 재귀 점화식 풀이 및 마스터 정리(Master Theorem)를 학습한다.
* [03. 억지 기법과 완전 탐색](./03.%20%EC%96%B5%EC%A7%80%20%EA%B8%B0%EB%B2%95%EA%B3%BC%20%EC%99%84%EC%A0%84%20%ED%83%90%EC%83%89.md) - Brute-force 알고리즘 설계 기법, 선택 정렬, 순차 탐색, 배낭 문제의 전수조사 및 문자열 단순 매칭을 인터랙티브 탐색 시뮬레이터와 함께 학습한다.
* [04. 축소 정복 기법](./04.%20%EC%B6%95%EC%86%8C%20%EC%A0%95%EB%B3%B5%20%EA%B8%B0%EB%B2%95.md) - Decrease-and-Conquer 알고리즘 설계 패러다임, 삽입 정렬, 위상 정렬(Topological Sort), 이진 탐색 및 가짜 동전 찾기 알고리즘을 학습한다.
* [05. 분할 정복 기법](./05.%20%EB%B6%84%ED%95%A0%20%EC%A0%95%EB%B3%B5%20%EA%B8%B0%EB%B2%95.md) - Divide-and-Conquer 기법의 핵심 원리, 병합 정렬(Merge Sort), 퀵 정렬(Quick Sort), 최근접 점의 쌍(Closest Pair) 및 스트라센 행렬 곱셈을 학습한다.
* [06. 공간으로 시간 벌기](./06.%20%EA%B3%B5%EA%B0%84%EC%9C%BC%EB%A1%9C%20%EC%8B%9C%EA%B0%84%20%EB%B2%8C%EA%B8%B0.md) - Space-and-Time Tradeoffs 기법, 카운팅 정렬, 해시 테이블의 충돌 해결 기법, 호스풀(Horspool) 문자열 매칭 알고리즘을 실시간 시뮬레이터와 함께 학습한다.
* [07. 동적 계획법](./07.%20%EB%8F%99%EC%A0%81%20%EA%B3%84%ED%9A%8D%EB%B2%95.md) - Dynamic Programming 패러다임, 메모이제이션과 바텀업 타뷸레이션, 0-1 배낭 문제, 최장 공통 부분 수열(LCS) 및 플로이드-워셜 최단 경로 알고리즘을 학습한다.
* [08. 탐욕적 기법](./08.%20%ED%83%90%EC%9A%95%EC%A0%81%20%EA%B8%B0%EB%B2%95.md) - Greedy 알고리즘의 설계 원리, 탐욕 선택 속성과 최적 부분 구조, 최소 신장 트리(크루스칼, 프림), 다익스트라 최단 경로 및 허프만 부호화 압축을 학습한다.
* [09. 백트래킹과 분기 한정](./09.%20%EB%B0%B1%ED%8A%B8%EB%9E%98%ED%82%B9%EA%B3%BC%20%EB%B6%84%EA%B8%B0%20%ED%95%9C%EC%A0%95.md) - 상태 공간 트리(State Space Tree) 탐색, 백트래킹의 깊이 우선 탐색 및 유망성 검사(Pruning), 분기 한정(Branch-and-Bound)의 최적 한계치 가지치기, N-Queen 문제를 학습한다.
* [10. NP-완전성과 근사 알고리즘](./10.%20NP-%EC%99%84%EC%A0%84%EC%84%B1%EA%B3%BC%20%EA%B7%BC%EC%82%AC%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98.md) - P, NP, NP-Complete, NP-Hard 복잡도 클래스, 다항 시간 환산(Polynomial-time Reduction) 및 NP-난해 문제 해결을 위한 근사 알고리즘을 학습한다.
* [algorithm-design-analysis 강의 흐름 지도](./00.%20algorithm-design-analysis%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 25개의 순서·쪽수·학습 점검을 연결한다.
