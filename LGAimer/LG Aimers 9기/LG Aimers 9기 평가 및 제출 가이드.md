---
aliases: []
course: lgaimer
created: '2026-02-03'
date: '2026-02-03'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: extracurricular
source: ''
status: draft
tags:
- ml
- extracurricular
- lecture
title: LG Aimers 9기 평가 및 제출 가이드
type: lecture
updated: '2026-05-05'
---

cohort:: [LG Aimers 9기](<./LG Aimers 9기.md>)

1. 리더 보드

### 평가 산식

최종 점수는 기본 모델 대비 성능 비율과 추론 시간 감소 비율의 가중 합산으로 계산됩니다.

$$Score = (Performance\ Norm \times 0.5) + (Speed\ Norm \times 0.5)$$

#### 1) 성능 비율 (Performance Norm)

기본 모델(**EXAONE-4.0-1.2B**)의 성능 대비 제출 모델의 성능 비율입니다.
$$Performance\ Norm = \frac{Performance_{target}}{Performance_{base}}$$

#### 2) 추론 시간 감소 비율 (Speed Norm)

기본 모델 대비 토큰당 추론 시간($T_{per\_token}$)의 감소 비율을 의미합니다.
$$Speed\ Norm = \frac{T_{base} - T_{target}}{T_{base}}$$

---

   해당 시간에는 모델 로드, 평가 데이터에 대한 추론, 결과값 파싱 등 모든 처리 과정이 포함되며, 이 전체 추론 실행 시간은 규칙으로 지정한 시간 제한(20분) 이내에 완료되어야 합니다.

   반면, 평가 산식에 반영되는 추론 시간은 모델이 평가용 벤치마크 데이터셋에 대해 실제로 추론을 수행한 시간만을 기준으로 측정한 '토큰 당 추론 시간'입니다.

Public score : 전체 테스트 데이터 100%
   ※ 테스트 데이터는 비공개 벤치셋으로 구성되어 있습니다.

1. 평가 방식
LG Aimers 수료 조건
Phase1을 이수하고 Phase2의 Public Score (LB: 0.5) 초과
기준 점수는 기본 모델(EXAONE-4.0-1.2B) 대비 성능·효율 개선을 반영한 기준
1차 평가 : 리더보드 Public Score 100%
동점자의 경우, 기존 리더보드 순위 산정 방식을 따름 [링크]의 '리더보드 점수' 부분을 참고
2차 평가 : 오프라인 해커톤(Phase3) 진출을 희망하는 팀은 코드 제출 후 코드 검증
Public 상위팀(약 100명)은 코드 및 PPT 필수 제출 대상
코드 제출과 검증를 모두 통과한 Public 상위팀(약 100명)이 오프라인 해커톤(Phase3) 진출

2. 코드 제출 대회 가이드
본 대회는 submit.zip 파일을 제출하는 방식의 '코드 제출 대회'로 진행됩니다.

참가자는 아래와 같은 구조로 submit.zip을 구성하여 제출해야 합니다.

아래의 구조와 동일하고 디렉토리 명과 파일 명을 모두 일치 시켜야합니다.

📁 참가자 제출 파일 구조 (submit.zip)

submit.zip
 └── model/        # 허깅 페이스(HF) 방식의 모델 가중치 파일 디렉토리
       └── (예: config.json, model.safetensors 등)
submit.zip 내 구조는 반드시 일치해야하며, 추가 최상위 폴더가 zip 구조 내 존재하는 경우 등 구조가 불일치하는 경우 설치 오류가 발생합니다.

⚙️ 평가 서버에서 추가되는 항목
제출 시, 평가 서버에서 참가자가 제출한 submit.zip 파일에는 아래 항목이 자동으로 추가됩니다.

submit.zip
├── model/        # 참가자 구성
├── script.py       # 평가에 사용될 추론 코드 (운영진 고정, 평가 시 자동 생성)
├── requirements.txt   # 평가에 사용될 환경 구성 (운영진 고정, 평가 시 자동 생성)
├── data/         # 평가에 사용될 테스트 데이터 (운영진 고정, 평가 시 자동 생성)
└── output/submission.csv        # 참가자 모델의 추론 결과가 저장되는 경로 (운영진 고정, 평가 시 자동 생성)
model/ 디렉토리만 참가자 제출 파일(zip)에서 구성할 수 있습니다.

💾 제출 파일 용량 제한

제출 파일(zip) 용량 제한: 최대 10GB 이내 (*압축해제 후 32GB 이내)

⏱️ 실행 시간 제한
전체 추론 코드 실행 시간: 최대 20분 이내 (시간 초과 시 제출 오류)

⚙️ 평가 서버 사양
OS : Ubuntu 22.04.5 LTS
CPU: 6 vCPU
RAM: 28GB
GPU : L4 (VRAM 22.4GiB)
Python : 3.11.14
CUDA : 12.8

💾 평가 서버 기본 설치 패키지(라이브러리) 목록

아래에 명시된 패키지(라이브러리)는 평가 서버에 사전 설치된 기본 환경이며, 참가자는 requirements.txt를 제출하거나 패키지 버전을 변경할 수 없습니다.
모든 제출물은 아래에 안내된 고정된 라이브러리 환경에서 그대로 실행·채점되며, 해당 환경과 호환되지 않는 코드 또는 다른 버전의 라이브러리를 전제로 한 구현은 실행 오류가 발생할 수 있습니다.
따라서, 참가자는 평가 서버에 기본 설치된 패키지(라이브러리 및 버전)를 기준으로 코드를 구현해야 하며, 아래 환경 외의 추가 라이브러리나 특정 버전에 의존하는 구현은 지원되지 않습니다.

1) 설치 패키지(라이브러리)

torch==2.9.0+cu128
numpy==2.2.6
pandas==2.3.3
transformers==4.57.3
tokenizers==0.22.1
accelerate==1.10.1
datasets==4.4.1
huggingface-hub==0.36.0
safetensors==0.7.0
sentencepiece==0.2.1
protobuf==6.33.2
evaluate==0.4.6
rouge_score==0.1.2
sacrebleu==2.5.1
tqdm==4.67.1
regex==2025.11.3
nltk==3.9.2
compressed-tensors==0.13.0
math-verify==0.8.0
antlr4-python3-runtime==4.11.0
sympy==1.14.0
langdetect==1.0.9
immutabledict==4.2.2
torch-c-dlpack-ext==0.1.4
vllm==0.14.1

1) 설치 시스템 패키지

python3.11
python3.11-distutils
python3-pip
python3.11-dev
build-essential
git
git-lfs
ninja-build
libomp-dev
libblas3
liblapack3
gfortran
libatlas-base-dev
curl
wget
ca-certificates
unzip
procps
cmake
tzdata
libxcb1

1) 모델 서빙 스펙

본 대회는 참가자가 제출한 EXAONE-4.0-1.2B 경량화 모델을 대상으로, 고정된 vLLM 서빙 환경에서 추론을 수행하여 평가합니다.
  ① 추론 엔진 (Inference Engine)
Inference Engine: vLLM (version: 0.14.1)
Model Interface: HuggingFace AutoModelForCausalLM 호환
제출된 모델은 다음 호출이 가능해야 하며, 토크나이저는 모델과 함께 제공되어야 합니다.

# tokenizer

AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)

# model

AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
  ② vLLM Serving 옵션
아래 옵션은 채점 서버에서 고정 적용됩니다.
tensor_parallel_size = 1
gpu_memory_utilization = 0.85
batch_size = auto
max_gen_toks = 16384
apply_chat_template = true

📌 유의사항
제출 시 발생하는 오류의 종류는 두 가지로 정의되며, 일일 제출 횟수 반영에 대한 기준이 다르므로 반드시 숙지하여 진행해야 합니다.

1) 설치 오류 : 제출하는 submit.zip 내부 구조가 불일치한 경우 -> 일일 제출 횟수 반영되지 않음
예) submit.zip 파일 안에 ./model 폴더로 되어 있지 않은 경우, 폴더명이 model이 아닌 경우
2) 제출 오류 : script.py 코드 실행 후 발생하는 모든 오류 -> 일일 제출 횟수 반영됨
예) tokenizer 로드가 안되는 경우, vLLM 모델 로딩에 실패한 경우, 전체 추론 시간이 20분을 초과하는 경우 등
평가 서버 환경은 인터넷 접속이 불가능하므로, 패키지 설치 이후 외부 다운로드가 필요한 코드나 모델은 작동하지 않습니다.
