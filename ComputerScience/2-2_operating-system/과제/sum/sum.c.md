---
aliases: []
course: operating-system
created: '2024-12-01'
date: '2024-12-01'
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/project
title: sum.c
type: project
updated: '2026-05-05'
---




up:: [[ComputerScience/2-2_operating-system/과제/SRTF/srtf.c|srtf.c]]
prerequisites:: [[ComputerScience/2-1_computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/2-1_linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/2-2_operating-system/과제/FCFS/fcfs.c|fcfs.c]], [[ComputerScience/2-2_operating-system/과제/Page/page.c|page.c]], [[ComputerScience/2-2_operating-system/과제/SJF/sjf.c|sjf.c]]

---
```c
#include <stdio.h>

int main() {
    FILE *fp, *fpOut;
    int i, j, k, min, max, n, sum;
  
    fp = fopen("sum.inp", "r");
  
    fpOut = fopen("sum.out", "w");
  
    fscanf(fp, "%d", &n);
  
    for (i = 0; i < n; i++) {
        fscanf(fp, "%d %d", &j, &k);
  
        if (j < k) {
            min = j;
            max = k;
        } else {
            min = k;
            max = j;
        }

        sum = 0;
        for (int x = min; x <= max; x++) {
            sum += x;
        }
  
        fprintf(fpOut, "%d\n", sum);
    }

    fclose(fp);
    fclose(fpOut);
  
    return 0;
}
```
