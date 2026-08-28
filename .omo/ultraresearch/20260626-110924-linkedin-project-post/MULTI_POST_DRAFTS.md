# LinkedIn Multi-Post Drafts

## 1. Portfolio: Verified Work Over Hype

AI 프로젝트들을 다시 정리하면서 기준이 하나 생겼습니다.

“무엇을 만들었는가”보다 “어디까지 다시 확인할 수 있는가”를 먼저 보자는 것.

UMMAYA는 공공서비스 요청을 작은 실행 단위와 권한 경계로 나누는 터미널 에이전트입니다. Hugging Face에는 GovOn/EXAONE 계열 모델과 민원·법률 응답 데이터셋을 올려 두었고, W&B에는 GovOn 실험 추적 프로젝트가 남아 있습니다.

DigitalPublishing에서는 WebGL 기반 모바일 청첩장 템플릿을 실제 배포 화면으로 확인했고, edu 아카이브에서는 CS/AI 학습과 LG Aimers 모델 효율성 학습을 지식그래프처럼 연결하고 있습니다.

이 작업들을 하나로 묶으면 “AI로 빨리 만들기”가 아니라 “검증 가능한 작업 흐름 만들기”에 더 가깝습니다.

앞으로도 포트폴리오를 결과물 목록이 아니라, 주장과 증거가 같이 남는 작업 기록으로 쌓아가려고 합니다.

여러분은 프로젝트를 공개할 때 어떤 증거를 가장 먼저 남기시나요?

본문 링크:
https://github.com/umyunsang

댓글에 둘 링크:
- UMMAYA docs: https://ummaya-docs.pages.dev/en/
- Hugging Face: https://huggingface.co/umyunsang
- W&B GovOn: https://wandb.ai/umyun3/GovOn
- DigitalPublishing demo: https://ourseason.pages.dev/
- edu archive: https://github.com/umyunsang/edu

#AI #SoftwareEngineering #Portfolio

## 2. UMMAYA: Civic Agent With Boundaries

공공서비스 AI를 만들 때 가장 위험한 문장은 “AI가 알아서 처리합니다”라고 생각합니다.

UMMAYA를 만들면서 잡은 방향은 반대입니다.

사용자는 자연어로 요청하지만, 시스템은 제한된 동작만 수행합니다. 찾기, 위치 확인, 검증, 제출, 문서화. 그리고 권한이 필요한 동작은 사용자의 동의와 경계가 먼저 드러나야 합니다.

이 프로젝트를 챗봇이 아니라 터미널 에이전트로 보는 이유도 여기에 있습니다. 답변을 그럴듯하게 만드는 것보다, 무엇을 실행했고 어디서 멈췄는지 보이게 만드는 일이 더 중요했습니다.

아직 alpha 단계의 프로젝트입니다. 그래서 더더욱 “공식 기관 시스템”처럼 보이게 쓰지 않으려고 합니다. UMMAYA가 보여주려는 것은 숨은 권한이 아니라, 제한된 실행과 명확한 handoff입니다.

공공서비스 AI에서 자동화보다 먼저 설계해야 하는 경계는 무엇일까요?

본문 링크:
https://ummaya-docs.pages.dev/en/

댓글에 둘 링크:
- Repo: https://github.com/umyunsang/UMMAYA
- Demo: https://github.com/umyunsang/UMMAYA/blob/main/assets/ummaya-demo.mp4
- Trust boundary docs: https://ummaya-docs.pages.dev/en/trust/what-ummaya-will-not-do/

#AgenticAI #GovTech #Trust

## 3. Hugging Face and W&B: Model Work as Trace

모델 실험을 공개할 때, 저는 “좋은 모델입니다”보다 “어떤 실험 흔적이 남아 있나요?”라는 질문이 더 중요하다고 느낍니다.

Hugging Face에는 GovOn/EXAONE 계열 실험을 공개해 두었습니다. 민원 응답용 QLoRA adapter, merged 모델, AWQ 모델, 민원·법률 응답 데이터셋, 그리고 실험용 Space들이 연결되어 있습니다.

W&B에는 `umyun3/GovOn` 프로젝트와 retrain, evaluation, hparam search 계열 프로젝트가 공개 메타데이터로 확인됩니다.

이걸 성능 자랑으로 쓰고 싶지는 않습니다. 아직 공개 포스트에서 말할 수 있는 것은 “민원/행정 도메인 모델을 실험했고, 모델·데이터셋·실험 추적 표면을 남겼다”는 정도입니다.

하지만 이 정도의 흔적만 있어도 다음 작업이 달라집니다.

어떤 데이터로 학습했는지, 어떤 형태의 모델로 배포 가능성을 봤는지, 어떤 실험이 실패했는지를 나중에 다시 확인할 수 있기 때문입니다.

여러분은 모델 실험을 공유할 때 성능 수치와 재현 흔적 중 무엇을 먼저 공개하시나요?

본문 링크:
https://huggingface.co/umyunsang/govon-civil-adapter

댓글에 둘 링크:
- Hugging Face profile: https://huggingface.co/umyunsang
- Civil response dataset: https://huggingface.co/datasets/umyunsang/govon-civil-response-data
- W&B GovOn: https://wandb.ai/umyun3/GovOn

#LLM #HuggingFace #MLOps

## 4. DigitalPublishing: Interaction as Publishing

WebGL 효과 하나를 모바일 청첩장 템플릿으로 바꾸면서 배운 것이 있습니다.

인터랙션은 장식이 아니라 읽는 순서를 설계하는 방식이라는 것.

DigitalPublishing의 모바일 초대장 작업은 Codrops의 이미지 unroll 레퍼런스를 바탕으로, 한국어 초대장 흐름에 맞게 스크롤, 이미지, 문구, 지도 연결, 마지막 reveal을 구성한 프로젝트입니다.

중요했던 건 “화려한 효과”가 아니었습니다.

모바일에서 첫 화면이 어떻게 보이는지, 이미지 전환이 텍스트를 방해하지 않는지, 실제 배포 URL에서 같은 경험이 유지되는지를 확인하는 일이 더 컸습니다.

소스코드가 통과해도 화면이 설득하지 못하면 퍼블리싱은 끝난 게 아니었습니다.

프론트엔드 작업에서 여러분은 어느 순간을 “완료”라고 보시나요? 코드가 준비됐을 때인가요, 실제 화면이 설득됐을 때인가요?

본문 링크:
https://ourseason.pages.dev/

댓글에 둘 링크:
- Project folder: https://github.com/umyunsang/DigitalPublishing/tree/main/mobile-wedding-unrolling-invitation
- DigitalPublishing repo: https://github.com/umyunsang/DigitalPublishing

#Frontend #WebGL #UX

## 5. edu and LG Aimers: Learning as Infrastructure

학습 기록은 쌓아두기만 하면 금방 무거워집니다.

그래서 `edu` 아카이브를 단순 폴더가 아니라, 다시 검색하고 연결할 수 있는 지식그래프처럼 관리하고 있습니다.

CS/AI 전공 학습, 알고리즘, 컴퓨터비전, 운영체제, 네트워크, LLM, AIOSS 흐름이 interface와 evidence 단위로 연결됩니다. LG Aimers 8기와 9기 자료도 같은 방식으로 묶었습니다.

특히 LG Aimers에서 남은 기준은 모델 효율성입니다.

정확도만 보는 것이 아니라, 실행 시간, serving 환경, vLLM/Hugging Face 호환성, 제한된 평가 시간 안에서 모델이 실제로 돌아가는지를 함께 봐야 했습니다.

이후 프로젝트를 볼 때도 그 기준이 계속 남습니다.

모델이 좋은가보다 먼저, 이 모델이 어떤 환경에서 돌아가며 실패했을 때 무엇을 남길 수 있는가를 보게 됩니다.

여러분은 학습 내용을 프로젝트로 다시 가져오기 위해 어떤 구조를 쓰시나요?

본문 링크:
https://github.com/umyunsang/edu

댓글에 둘 링크:
- LG Aimers 8기 note: https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%208%EA%B8%B0/LG%20Aimers%208%EA%B8%B0.md
- LG Aimers 9기 note: https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%209%EA%B8%B0/LG%20Aimers%209%EA%B8%B0.md

#ComputerScience #LearningArchive #MLOps

## 6. IlluOps: Evidence Before Claims

AI 작업을 하다 보면 결과물보다 먼저 봐야 하는 것이 있습니다.

이 결과가 어디까지 검증됐는가.

IlluOps는 아직 완성된 앱으로 말할 수 있는 프로젝트가 아닙니다. public repo도 전체 구현체가 아니라 planning, reference, evidence boundary를 보여주는 표면에 가깝습니다.

그래서 이 프로젝트를 설명할 때 일부러 조심스럽게 씁니다.

“출시했다”가 아니라 “어떤 주장을 어떤 근거와 연결할지 정리하고 있다.”

“검증됐다”가 아니라 “공개 링크로 증명할 수 있는 것과 로컬 검증에만 남겨야 하는 것을 분리하고 있다.”

AI가 만든 결과는 그럴듯할 수 있습니다. 하지만 제품이나 포트폴리오로 남기려면, 주장과 증거의 거리가 짧아야 합니다.

저는 이 기준이 앞으로 더 중요해질 것 같습니다.

여러분은 AI-assisted work를 공개할 때 어떤 검증 기준을 최소선으로 두시나요?

본문 링크:
https://github.com/umyunsang/IlluOps

댓글에 둘 링크:
- Claim support matrix: https://github.com/umyunsang/IlluOps/blob/main/references/claim_support_matrix.tsv

#AIWorkflow #QualityAssurance #Engineering
