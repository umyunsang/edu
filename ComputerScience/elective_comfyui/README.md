---
aliases: []
course: comfyui
created: '2026-03-19'
date: '2026-03-19'
semester: elective
source: ''
status: evergreen
tags:
- cs/ai
- cs/cv
- type/index
title: 🎨 ComfyUI & Generative AI Workflows
type: index
updated: '2026-05-05'
---

# 🎨 ComfyUI & Generative AI Workflows

<p align="center">
  <a href="https://huggingface.co/umyunsang"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge"></a>
  <img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-Node--Based%20GUI-7C3AED?style=for-the-badge&logo=comfyui&logoColor=white">
</p>

이 섹션은 **ComfyUI**를 활용하여 Stable Diffusion 모델에 LoRA를 적용하고, 다양한 커스텀 노드를 연결하여 구축한 나만의 생성형 AI 워크플로우 실습 기록입니다.

## 🚀 주요 실습 내용

### 1. LoRA Application & Model Tuning
- **Hugging Face 모델 공유**: [umyunsang Hugging Face 🔗](https://huggingface.co/umyunsang)
- **학습 도구**: [OneTrainer 🔗](https://github.com/umyunsang/OneTrainer-fork) (Stable Diffusion 모델 학습 및 튜닝)
- 직접 학습하거나 튜닝한 LoRA 모델을 ComfyUI 워크플로우에 통합하여 특정 스타일이나 캐릭터를 구현하는 실습을 진행했습니다.

### 2. Custom Node Workflows
- **Node-Based Architecture**: 복잡한 생성 과정을 노드 단위로 분해하고 연결하여 자동화된 파이프라인 구축
- **워크플로우 관리**: [ComfyUI-Manager 🔗](https://github.com/umyunsang/ComfyUI-Manager)를 통한 커스텀 노드 설치 및 종속성 관리
- **주요 커스텀 노드 활용**:
    - ControlNet 연동을 통한 포즈 및 구도 제어
    - Upscaler 노드를 활용한 고해상도 이미지 생성
    - IPAdapter를 이용한 이미지 기반 스타일 전이

## 📂 관련 리소스
- [Hugging Face Profile](https://huggingface.co/umyunsang)

---
마지막 업데이트: 2026-03-19

## All notes in this course (auto)
```dataview
TABLE status, file.mtime as updated
FROM "ComputerScience/elective_comfyui"
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
```
