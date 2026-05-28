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
title: fcfs.c
type: project
updated: '2026-05-05'
---




up:: [[ComputerScience/2-2_operating-system/과제/Banker/banker.c|banker.c]]
prerequisites:: [[ComputerScience/2-1_computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/2-1_linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/2-2_operating-system/과제/Page/page.c|page.c]], [[ComputerScience/2-2_operating-system/과제/SJF/sjf.c|sjf.c]], [[ComputerScience/2-2_operating-system/과제/SRTF/srtf.c|srtf.c]]

---
```c
#include <stdio.h>
  
typedef struct {
    int arrival_time;
    int cpu_time;
    int waiting_time;
} Process;
  
int main() {
    FILE *input_file = fopen("fcfs.inp", "r");
    FILE *output_file = fopen("fcfs.out", "w");
  
    if (!input_file || !output_file) {
        perror("File opening failed");
        return 1;
    }
  
    int n;
    fscanf(input_file, "%d", &n);
    Process processes[n];
  
    for (int i = 0; i < n; i++) {
        fscanf(input_file, "%d %d", &processes[i].arrival_time, &processes[i].cpu_time);
        processes[i].waiting_time = 0;
    }
  
    int current_time = 0;
    int total_waiting_time = 0;
  
    // FCFS 스케줄링
    for (int i = 0; i < n; i++) {
        if (current_time < processes[i].arrival_time) {
            current_time = processes[i].arrival_time;
        }
        processes[i].waiting_time = current_time - processes[i].arrival_time;
        total_waiting_time += processes[i].waiting_time;
        current_time += processes[i].cpu_time;
    }
  
    fprintf(output_file, "%d\n", total_waiting_time);
    fclose(input_file);
    fclose(output_file);
    return 0;
}
```
