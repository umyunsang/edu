---
aliases:
- LG Aimers 9기 강의자료 원소스
course: lgaimer
created: '2026-06-25'
date: '2026-06-25'
semester: extracurricular
source: https://academy.lgresearch.ai/study
status: stable
tags:
- ml
- source-index
- academy
title: LG Aimers 9기 강의자료 원소스
type: source-index
updated: '2026-06-25'
---

course:: [LG Aimers 9기](<./LG Aimers 9기.md>)
materials_path:: `강의자료/`
public_curriculum:: [LG Aimers](https://www.lgaimers.ai/)
academy_study:: [LG AI ACADEMY study](https://academy.lgresearch.ai/study)
research_journal:: 20260625-182513

이 문서는 LG Aimers 9기 강의자료의 로컬 원소스 기록입니다. 실제 다운로드 링크는 `https://academy.lgresearch.ai/study`의 로그인 세션 뒤에 있으며, 2026-06-25 18:32 KST 기준 비인증 접근은 `/login`으로 이동하거나 `RT_AUTHENTICATION_FAILURE`를 반환했습니다.

## 확인 기준

- 공식 공개 커리큘럼: `https://www.lgaimers.ai/`
- 로그인 학습 페이지: `https://academy.lgresearch.ai/study`
- 로컬 자료 폴더: `강의자료/`
- 연구 감사 로그: `.omo/ultraresearch/20260625-182513/`

## 공식 커리큘럼 대조

| 모듈 | 교수 | 로컬 자료 상태 |
| --- | --- | --- |
| Tabular ML: From Classical Models to Foundation Models | 이한국, 성균관대학교 소프트웨어학과 | 노트북 1개 사용 가능, PDF 6개는 Git LFS pointer |
| Optimization and Decision-Focused Learning / Time-Series Analysis | 이용재, UNIST 산업공학과 | PDF 2개는 Git LFS pointer |
| 딥러닝 자연어처리 기초와 LLM Agent | 이환희, 중앙대학교 AI 학과 | PDF 사용 가능 |
| Mathematics for ML | 신진우, KAIST 전기및전자공학부 | PDF 사용 가능 |
| LLM Application & Evaluation | 김재형, 연세대학교 첨단컴퓨팅학부 | PDF 사용 가능 |
| 지도학습 | 노알버트, 연세대학교 인공지능학과 | PDF 사용 가능 |

## 로컬 자료 인벤토리

| 상태 | 자료 | 검증 |
| --- | --- | --- |
| 사용 가능 | Mathematics for ML | PDF, 82 pages, sha256 `a9fc42785cbdce40f04078747b464de0a66bda52fd132f7e4091db13b69dc416` |
| 사용 가능 | 지도학습 | PDF, 156 pages, sha256 `714c1e2d590436c085f88e1b61da73976f8b64af240b67126a8280090fccfa83` |
| 사용 가능 | LLM Application & Evaluation | PDF, 196 pages, sha256 `d099a41ec8e6b9a95efa6f516ec37486f8ebe1ea3f1a6ed48437943f5b330e4e` |
| 사용 가능 | 딥러닝 자연어처리 기초와 LLM Agent | PDF, 204 pages, sha256 `491286521f3cefdbaef94fb2ac2f375bf34ebb0974b6b0842f1c0dcfe92de042` |
| 사용 가능 | Hands-on Tabular ML | Notebook, nbformat 4, 37 cells, sha256 `e5b8969fa3ead5b8d041ce95ae2ec72befac4f776ba085d7eaaca1f41fb33482` |
| 미복원 | Introduction to Tabular ML | Git LFS pointer, expected size 618582, oid `98a3a0281f82f515d1f6683409a7456a75cf84ea8bc561a85effaabebed5e27d` |
| 미복원 | Classical ML for Tabular Data | Git LFS pointer, expected size 2278574, oid `4e44d963f4186e8eb886232b11137159f440e08266af6767a8cde10c612b2a44` |
| 미복원 | Deep Architectures for Tabular Data | Git LFS pointer, expected size 4638876, oid `f733074ad69dc6d2b2a5c92d45743e1ea06a0726ce8d686a6effe9350753a265` |
| 미복원 | Tabular Representation Learning | Git LFS pointer, expected size 2944923, oid `a61634827998381b0ec7cf491a26c16ed7df96594e52d7e0d6a50daed9f2d9dc` |
| 미복원 | LLMs with Tabular Data | Git LFS pointer, expected size 5971928, oid `e5ec8e255f28681ec010082efce8ec391551998d605d736cbd0aa306b3de49e8` |
| 미복원 | A New Paradigm: TabPFN | Git LFS pointer, expected size 4604735, oid `2baa9c9a310545d6bd402f6e7de839cb52499574f28d0b3da931e741cabce5da` |
| 미복원 | 이용재 교수 1-3강: Opt & DFL | Git LFS pointer, expected size 3694954, oid `ad4ef409ab24ce7bd66d5a147251cbdb3a261dc0c4cc7dbe8dd844714ea349ee` |
| 미복원 | 이용재 교수 4-6강: Time Series | Git LFS pointer, expected size 5167681, oid `0b28fc37ecbe2f4e60c7825413e0fafd122a125cb63696a450b8ab91aedd6ca5` |

## 업스트림 접근 기록

- `https://academy.lgresearch.ai/study`는 HTML shell을 반환하지만 브라우저 렌더링 후 `/login`으로 이동했습니다.
- 공개 HTML은 Create React App shell이고 `/static/js/main.400d8c09.js`, `/static/css/main.5ab3849a.css`를 참조합니다.
- `https://academy.lgresearch.ai/asset-manifest.json`에서 112개 JS asset을 확인했고, 학습/콘텐츠 관련 lazy chunk를 저장했습니다.
- 학습 관련 API 후보는 `/api/v1/portal/courses/my-courses`, `/api/v1/portal/courses/`, `/api/v1/portal/courses/main/`, `/api/v1/portal/contents/`, `/api/v1/common/files/{fileId}/signed-url`입니다.
- 비인증 API 확인 결과 `/api/v1/auth-me`, `/api/v1/portal/courses/my-courses`, `/api/v1/portal/contents/`, `/api/v1/common/files/1/signed-url`은 401 `RT_AUTHENTICATION_FAILURE`를 반환했습니다.

## 보완 필요

- 로그인된 academy 세션에서 `/study`의 실제 courseId, contentId, fileId, signed URL을 캡처해야 합니다.
- Git LFS pointer 8개는 현재 파일명이 PDF여도 PDF 본문이 아니므로 복원 전까지 학습자료로 간주하지 않습니다.
- 이 폴더는 현재 Git repository 내부로 인식되지 않아 `git lfs pull`로 즉시 복원할 수 없었습니다.
