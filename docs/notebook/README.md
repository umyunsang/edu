# 노트북 근거 데이터

quantum-ml 실습 노트북 174개의 출처·품질 기록. 정리문서에서 노트북을 인용할 때
근거로 삼는다. 원본은 `.omo/drive-sync/` 에 있으나 그쪽은 git 추적 대상이 아니다.

| 파일 | 내용 |
| :-- | :-- |
| `notebook-evidence.json` | 노트북 174개 메타. `fidelity` 로 인용 가능 여부가 갈리고, `outputs.figure_cells` 에 그림이 있는 셀 번호가 들어 있다. |
| `notebook-rename-map.json` | 재네이밍 구→신 경로 174건. 이름이 바뀌기 전에 만들어진 산출물과 대조할 때 쓴다. |
| `drive-manifest.json` | Google Drive 원본 py ↔ 노트북 대응. `raw_path` ↔ `artifact_path`. |
| `derived-artifacts.json` | 원본 없이 가공된 노트북 12건의 출처 기록. |

## fidelity 값

| 값 | 수 | 인용 |
| :-- | --: | :-- |
| `faithful` | 167 | 가능 — 원본 py 라인을 전량 보존 |
| `expression_differs` | 4 | 가능 — 원본 요소 100% 유지, 표현만 다름 |
| `rewritten` | 1 | **금지** — 원본에 없는 코드가 추가됨 |
| `support_module` | 2 | 실습이 아니라 공통 라이브러리 |

`rewritten` 1건은 `work/01_iris_environment_eda_logistic_knn_baselines.ipynb` 다.
원본 `Day1/1-1.py` 에 없는 KNeighbors · StandardScaler · make_pipeline ·
ConfusionMatrix · stratify · classification_report 가 들어갔고 수치도 다르다
(`test_size` 0.2→0.25, `max_iter` 300→200). "강의에서 다룬 코드"로 인용하면 사실과 어긋난다.
