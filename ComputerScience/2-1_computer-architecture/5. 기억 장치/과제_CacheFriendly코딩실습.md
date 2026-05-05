---
aliases: []
course: computer-architecture
created: '2024-05-26'
date: '2024-05-26'
semester: 2-1
source: ''
status: seedling
tags:
- type/lecture
title: 과제_CacheFriendly코딩실습
type: lecture
updated: '2026-05-05'
---

---

#### cache_hit

```c
#include <stdio.h>
#include <time.h>

int main(){
	int i, j, sum = 0;
	clock_t start = clock();

	static int x[4000][4000];
	for(i = 0; i < 4000; i++) {
		for(j = 0; j < 4000; j++) {
			sum += x[i][j]
		}
	}

	clock_t end = clock();
	printf("소요시간 : %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}

```

#### cache_miss

```c
#include <stdio.h>
#include <time.h>

int main(){
	int i, j, sum = 0;
	clock_t start = clock();

	static int x[4000][4000];
	for(j = 0; j < 4000; j++) {
		for(i = 0; i < 4000; i++) {
			sum += x[i][j]
		}
	}

	clock_t end = clock();
	printf("소요시간 : %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}
```
