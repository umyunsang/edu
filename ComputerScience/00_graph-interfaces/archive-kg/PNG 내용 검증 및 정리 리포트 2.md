---
aliases: []
course: archive-kg
created: '2026-05-28'
date: '2026-05-28'
semester: meta
source: ''
status: evergreen
tags:
- type/interface
- pkm/kg-evidence
title: PNG 내용 검증 및 정리 리포트
type: interface
updated: '2026-05-28'
---

kg_skeleton:: [[ComputerScience/00_graph-interfaces/archive-kg/2026 GraphRAG 아카이브 스켈레톤|2026 GraphRAG 아카이브 스켈레톤]]

# PNG 내용 검증 및 정리 리포트

PNG 파일을 OCR, 크기, exact SHA-256 duplicate, 현재 embed 참조, 과목 prefix/노트 token overlap으로 검증했습니다.

## Summary

- Initial PNG files: 1670
- Exact duplicate files deleted: 80
- Duplicate reference rewrite passes: 88
- Orphan PNGs OCR-scanned: 167
- Orphan PNGs semantically renamed: 36
- Embedded into source notes: 71
- Course media indexes written: 4
- Unmatched orphan PNGs deleted: 73
- Remaining PNG files: 1517
- Remaining unembedded PNG files: 0

## Deleted exact duplicates

- `image/3-1_distributed-computing__Pasted image 20250326124513.png -> image/3-1_distributed-computing__Pasted image 20250326124514.png`
- `image/3-1_distributed-computing__Pasted image 20250326124557.png -> image/3-1_distributed-computing__Pasted image 20250326124558.png`
- `image/3-1_intellectual-property__Pasted image 20250326093234.png -> image/3-1_intellectual-property__Pasted image 20250326093233.png`
- `image/3-1_intellectual-property__Pasted image 20250326093245.png -> image/3-1_intellectual-property__Pasted image 20250326093244.png`
- `image/3-1_intellectual-property__Pasted image 20250430090937.png -> image/3-1_intellectual-property__Pasted image 20250430090936.png`
- `image/3-1_machine-learning__Pasted image 20250327165760.png -> image/3-1_machine-learning__Pasted image 20250327165759.png`
- `image/3-1_machine-learning__Pasted image 20250329135532.png -> image/3-1_machine-learning__Pasted image 20250329135531.png`
- `image/3-1_machine-learning__Pasted image 20250415173230.png -> image/3-1_machine-learning__Pasted image 20250415173231.png`
- `image/3-1_machine-learning__Pasted image 20250415173443.png -> image/3-1_machine-learning__Pasted image 20250415173442.png`
- `image/3-1_machine-learning__Pasted image 20250415173547.png -> image/3-1_machine-learning__Pasted image 20250415173546.png`
- `image/3-1_machine-learning__Pasted image 20250417161009.png -> image/3-1_machine-learning__Pasted image 20250417161611.png`
- `image/3-1_machine-learning__Pasted image 20250417161010.png -> image/3-1_machine-learning__Pasted image 20250417161611.png`
- `image/3-1_machine-learning__Pasted image 20250417161612.png -> image/3-1_machine-learning__Pasted image 20250417161611.png`
- `image/3-1_machine-learning__Pasted image 20250417161203.png -> image/3-1_machine-learning__Pasted image 20250417161202.png`
- `image/3-1_machine-learning__Pasted image 20250421120642.png -> image/3-1_machine-learning__Pasted image 20250421120641.png`
- `image/3-1_machine-learning__Pasted image 20250421120854.png -> image/3-1_machine-learning__Pasted image 20250421120853.png`
- `image/3-1_machine-learning__Pasted image 20250421121449.png -> image/3-1_machine-learning__Pasted image 20250421121448.png`
- `image/3-1_machine-learning__Pasted image 20250421121942.png -> image/3-1_machine-learning__Pasted image 20250421121943.png`
- `image/3-1_programming-languages__Pasted image 20250326155553.png -> image/3-1_programming-languages__Pasted image 20250326155552.png`
- `image/4-1_algorithm__dp-grid3-page24-k2-solution-annotated-v3.png -> image/4-1_algorithm__dp-grid3-page24-k2-solution-annotated-v2.png`
- `image/4-1_algorithm__dp-grid3-pdf-conditions-annotated-v4.png -> image/4-1_algorithm__dp-grid3-pdf-conditions-annotated-v5.png`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v8.png -> image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v3.png`
- `image/Pasted image 20240511133103.png -> image/Pasted image 20240511133126.png`
- `image/Pasted image 20240511135852.png -> image/Pasted image 20240511135940.png`
- `image/Pasted image 20240516150328.png -> image/Pasted image 20240516150307.png`
- `image/Pasted image 20240516150330.png -> image/Pasted image 20240516150307.png`
- `image/Pasted image 20240520184104.png -> image/Pasted image 20240520184106.png`
- `image/Pasted image 20240520184949.png -> image/Pasted image 20240520184959.png`
- `image/Pasted image 20240520185329.png -> image/Pasted image 20240520185332.png`
- `image/Pasted image 20240520192459.png -> image/Pasted image 20240520192500.png`
- `image/Pasted image 20240520195647.png -> image/Pasted image 20240520195649.png`
- `image/Pasted image 20240523165210.png -> image/Pasted image 20240523165241.png`
- `image/Pasted image 20240525112214.png -> image/Pasted image 20240525112222.png`
- `image/Pasted image 20240525155948.png -> image/Pasted image 20240525160001.png`
- `image/Pasted image 20240530104048.png -> image/Pasted image 20240530104050.png`
- `image/Pasted image 20240530112124.png -> image/Pasted image 20240530112128.png`
- `image/Pasted image 20240530114653.png -> image/Pasted image 20240530114656.png`
- `image/Pasted image 20240612152332.png -> image/Pasted image 20240612152339.png`
- `image/Pasted image 20240911114002.png -> image/Pasted image 20240911113952.png`
- `image/Pasted image 20240911172915.png -> image/Pasted image 20240911172632.png`
- `image/Pasted image 20240920133551.png -> image/Pasted image 20240920133558.png`
- `image/Pasted image 20240923092655.png -> image/Pasted image 20240923092625.png`
- `image/Pasted image 20240923170312.png -> image/Pasted image 20240923170316.png`
- `image/Pasted image 20240923171706.png -> image/Pasted image 20240923171709.png`
- `image/Pasted image 20240926104730.png -> image/Pasted image 20240926103203.png`
- `image/Pasted image 20240926104735.png -> image/Pasted image 20240926103203.png`
- `image/Pasted image 20241011153825.png -> image/Pasted image 20241011153812.png`
- `image/Pasted image 20241015140122.png -> image/Pasted image 20241015140156.png`
- `image/Pasted image 20241031110646.png -> image/Pasted image 20241031110649.png`
- `image/Pasted image 20241031111030.png -> image/Pasted image 20241031111033.png`
- `image/Pasted image 20241104152347.png -> image/Pasted image 20241104151814.png`
- `image/Pasted image 20241118163119.png -> image/Pasted image 20241118163121.png`
- `image/Pasted image 20241118170034.png -> image/Pasted image 20241118170035.png`
- `image/Pasted image 20241118170829.png -> image/Pasted image 20241118170323.png`
- `image/Pasted image 20241122154607.png -> image/Pasted image 20241122154958.png`
- `image/Pasted image 20241124152253.png -> image/Pasted image 20241124152258.png`
- `image/Pasted image 20241125105754 1.png -> image/Pasted image 20241125105757.png`
- `image/Pasted image 20241125105754.png -> image/Pasted image 20241125105757.png`
- `image/Pasted image 20241207134248.png -> image/Pasted image 20241207112943.png`
- `image/Pasted image 20241217152541.png -> image/Pasted image 20241217152556.png`
- `image/Pasted image 20241217152544.png -> image/Pasted image 20241217152556.png`
- `image/Pasted image 20241217152550.png -> image/Pasted image 20241217152556.png`
- `image/Pasted image 20241217152555.png -> image/Pasted image 20241217152556.png`
- `image/Pasted image 20241217153001.png -> image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153004.png -> image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153006.png -> image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153008.png -> image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153011.png -> image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153120.png -> image/Pasted image 20241217153127.png`
- `image/Pasted image 20241217153122.png -> image/Pasted image 20241217153127.png`
- `image/Pasted image 20241217153123.png -> image/Pasted image 20241217153127.png`
- `image/Pasted image 20241217153124.png -> image/Pasted image 20241217153127.png`
- `image/Pasted image 20241217153126.png -> image/Pasted image 20241217153127.png`
- `image/Pasted image 20250919184557.png -> image/Pasted image 20250919184828.png`
- `image/elective_LLM__Pasted image 20250120103622.png -> image/elective_LLM__Pasted image 20250120103621.png`
- `image/elective_LLM__Pasted image 20250120103626.png -> image/elective_LLM__Pasted image 20250120103621.png`
- `image/elective_LLM__Pasted image 20250120103633.png -> image/elective_LLM__Pasted image 20250120103621.png`
- `image/elective_LLM__Pasted image 20250120103634.png -> image/elective_LLM__Pasted image 20250120103621.png`
- `image/elective_LLM__Pasted image 20250120111924.png -> image/elective_LLM__Pasted image 20250120111923.png`
- `image/elective_LLM__Pasted image 20250121150552.png -> image/elective_LLM__Pasted image 20250121150551.png`

## Semantically renamed orphan PNGs

- `image/Pasted image 20240516150732.png -> image/computer-networks__Routing Information Protocol (RIP)__- left none get b.png`
- `image/Pasted image 20240520183032.png -> image/computer-architecture__3. 레지스터__BBo Als OS Sat HO AS St.png`
- `image/Pasted image 20240520191325.png -> image/computer-architecture__3. 명령어 사이클__oe! mee Subt o e e.png`
- `image/Pasted image 20240523153919.png -> image/computer-networks__Routing Information Protocol (RIP)__ptelstens SENSES.png`
- `image/Pasted image 20240523153946.png -> image/computer-networks__Routing Information Protocol (RIP)__H 18 M Pivot $= 0.png`
- `image/Pasted image 20240604134248.png -> image/artificial-intelligence__ResNet__Convolution layer.png`
- `image/Pasted image 20240610191100.png -> image/computer-networks__Routing Information Protocol (RIP)__Pcxly = 0) Pcxly.png`
- `image/Pasted image 20240610191125.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi.png`
- `image/Pasted image 20240610191214.png -> image/computer-networks__Routing Information Protocol (RIP)__xX it X 1.png`
- `image/Pasted image 20240711153208.png -> image/computer-networks__Routing Information Protocol (RIP)__Ao 6-14 WERBZIO B.png`
- `image/Pasted image 20240711153349.png -> image/computer-networks__Routing Information Protocol (RIP)__Aol 6-6 7PM a.png`
- `image/Pasted image 20240812142815.png -> image/computer-networks__Routing Information Protocol (RIP)__o https github.co.png`
- `image/Pasted image 20240909173818.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 2.png`
- `image/Pasted image 20240911172611.png -> image/computer-networks__Routing Information Protocol (RIP)__of GIOLE, C1.png`
- `image/Pasted image 20240920132218.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 3.png`
- `image/Pasted image 20240920133516.png -> image/computer-networks__Routing Information Protocol (RIP)__CRE) 22 sxe sete.png`
- `image/Pasted image 20240927153310.png -> image/computer-networks__Routing Information Protocol (RIP)__Example.png`
- `image/Pasted image 20241010103803.png -> image/database-systems__데이터 베이스 언어 SQL__CREATE TABLE & 0 S_01S.png`
- `image/Pasted image 20241107171107.png -> image/computer-networks__Routing Information Protocol (RIP)__Routing Procotol.png`
- `image/Pasted image 20241108153037.png -> image/computer-networks__Routing Information Protocol (RIP)__B2 4-3 Of8 ZH.png`
- `image/Pasted image 20241118165250.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 4.png`
- `image/Pasted image 20241118165308.png -> image/computer-networks__Routing Information Protocol (RIP)__ARORA Senay.png`
- `image/Pasted image 20241122152122.png -> image/computer-networks__Routing Information Protocol (RIP)__7,2 1,21 24 (comp.png`
- `image/Pasted image 20241122155047.png -> image/computer-networks__Routing Information Protocol (RIP)__Let A= {0,1,2,3}.png`
- `image/Pasted image 20241122160849.png -> image/computer-networks__Routing Information Protocol (RIP)__Ol 4-12 SBPaSt OF.png`
- `image/Pasted image 20241127171952.png -> image/computer-networks__기말 암기 정리__213-2 $2 well-known ZE Wis.png`
- `image/Pasted image 20241129152258.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 5.png`
- `image/Pasted image 20241129153104.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 6.png`
- `image/Pasted image 20241210163011.png -> image/computer-networks__네트워크 계층 작업과 프로토콜__MAC RS AB.COEF gy MAC 32 A,B.C.D.E.png`
- `image/Pasted image 20241213124614.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 7.png`
- `image/Pasted image 20241215151921.png -> image/computer-networks__Routing Information Protocol (RIP)__HOHHOAS AS SAO SA.png`
- `image/Pasted image 20250904151517.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 8.png`
- `image/Pasted image 20250919124246.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 9.png`
- `image/Pasted image 20250919184828.png -> image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 10.png`
- `image/스크린샷 2024-11-22 150802.png -> image/computer-networks__Routing Information Protocol (RIP)__ae 47 HAlet Ze.png`
- `image/스크린샷 2024-11-22 150830.png -> image/computer-networks__Routing Information Protocol (RIP)__B2 42 SBHS 01st D.png`

## Embedded into source notes

- `image/3-1_distributed-computing__Pasted image 20250326125902.png -> ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism.md score=7`
- `image/3-2_bigdata-analysis__image.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=21`
- `image/3-2_bigdata-analysis__스크린샷 2025-09-17 15.38.05.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark.md score=10`
- `image/3-2_optimization-math__KakaoTalk_Photo_2025-10-23-12-58-48 001.png -> ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이.md score=5`
- `image/3-2_optimization-math__KakaoTalk_Photo_2025-10-23-12-58-49 003.png -> ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix.md score=10`
- `image/3-2_optimization-math__KakaoTalk_Photo_2025-10-23-12-58-49 004.png -> ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix.md score=10`
- `image/4-1_algorithm__dp-edit-02-operations.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=5`
- `image/4-1_algorithm__dp-edit-03-trace.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=5`
- `image/4-1_algorithm__dp-floyd-03-path.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=6`
- `image/4-1_algorithm__dp-floyd-05-path-recovery.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=6`
- `image/4-1_algorithm__dp-grid-01-flow.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=5`
- `image/4-1_algorithm__dp-grid3-pdf-conditions-annotated-v3.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=7`
- `image/4-1_algorithm__dp-grid3-pdf-k1-state-annotated-v2.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=7`
- `image/4-1_algorithm__dp-grid3-pdf-k1-state-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__dp-grid3-pdf-variants-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__dp-knapsack-02-cell.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=5`
- `image/4-1_algorithm__dp-knapsack-03-trace.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=7`
- `image/4-1_algorithm__dp-knapsack-pdf-cases-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=9`
- `image/4-1_algorithm__dp-knapsack-pdf-table-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__dp-lcs-02-cell.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=5`
- `image/4-1_algorithm__dp-lcs-04-find-lcs.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=7`
- `image/4-1_algorithm__dp-lcs-supp50-find-an-lcs-annotated-v3.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=9`
- `image/4-1_algorithm__dp-matrix-01-structure.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=8`
- `image/4-1_algorithm__dp-matrix-02-cost.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=5`
- `image/4-1_algorithm__dp-matrix-04-supp-sixchain.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=10`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v4.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=6`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v5.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=10`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v6.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=11`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated-v7.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__dp-matrix-pdf-final-order-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=8`
- `image/4-1_algorithm__greedy-coin-pdf-page04-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__greedy-coin-pdf-page06-counterexample-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=6`
- `image/4-1_algorithm__greedy-fractional-pdf-page08-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=6`
- `image/4-1_algorithm__greedy-prim-pdf-page11-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리.md score=6`
- `image/4-1_algorithm__greedy-prim-pdf-page13-dist-annotated.png -> ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md score=14`
- `image/computer-networks__Routing Information Protocol (RIP)__- left none get b.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=6`
- `image/computer-architecture__3. 레지스터__BBo Als OS Sat HO AS St.png -> ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터.md score=9`
- `image/computer-architecture__3. 명령어 사이클__oe! mee Subt o e e.png -> ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클.md score=11`
- `image/computer-networks__Routing Information Protocol (RIP)__ptelstens SENSES.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=9`
- `image/computer-networks__Routing Information Protocol (RIP)__H 18 M Pivot $= 0.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=7`
- `image/artificial-intelligence__ResNet__Convolution layer.png -> ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet.md score=8`
- `image/computer-networks__Routing Information Protocol (RIP)__Pcxly = 0) Pcxly.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=5`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=5`
- `image/computer-networks__Routing Information Protocol (RIP)__xX it X 1.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=7`
- `image/computer-networks__Routing Information Protocol (RIP)__Ao 6-14 WERBZIO B.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=8`
- `image/computer-networks__Routing Information Protocol (RIP)__Aol 6-6 7PM a.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=6`
- `image/computer-networks__Routing Information Protocol (RIP)__o https github.co.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=43`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 2.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=6`
- `image/computer-networks__Routing Information Protocol (RIP)__of GIOLE, C1.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=10`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 3.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=7`
- `image/computer-networks__Routing Information Protocol (RIP)__CRE) 22 sxe sete.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=9`
- `image/computer-networks__Routing Information Protocol (RIP)__Example.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=11`
- `image/database-systems__데이터 베이스 언어 SQL__CREATE TABLE & 0 S_01S.png -> ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL.md score=15`
- `image/computer-networks__Routing Information Protocol (RIP)__Routing Procotol.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=19`
- `image/computer-networks__Routing Information Protocol (RIP)__B2 4-3 Of8 ZH.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=7`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 4.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=6`
- `image/computer-networks__Routing Information Protocol (RIP)__ARORA Senay.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=5`
- `image/computer-networks__Routing Information Protocol (RIP)__7,2 1,21 24 (comp.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=8`
- `image/computer-networks__Routing Information Protocol (RIP)__Let A= {0,1,2,3}.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=6`
- `image/computer-networks__Routing Information Protocol (RIP)__Ol 4-12 SBPaSt OF.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=8`
- `image/computer-networks__기말 암기 정리__213-2 $2 well-known ZE Wis.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리.md score=8`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 5.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=6`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 6.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=7`
- `image/computer-networks__네트워크 계층 작업과 프로토콜__MAC RS AB.COEF gy MAC 32 A,B.C.D.E.png -> ComputerScience/04_systems-infrastructure/computer-networks/12. 네트워크 계층 작업과 프로토콜/네트워크 계층 작업과 프로토콜.md score=5`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 7.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=5`
- `image/computer-networks__Routing Information Protocol (RIP)__HOHHOAS AS SAO SA.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=5`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 8.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=8`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 9.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=8`
- `image/big-data-analysis__BDA_Hands_on_Numerical_and_Textual_Data_Analytics_usi 10.png -> ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API.md score=10`
- `image/computer-networks__Routing Information Protocol (RIP)__ae 47 HAlet Ze.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=6`
- `image/computer-networks__Routing Information Protocol (RIP)__B2 42 SBHS 01st D.png -> ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP).md score=5`

## Deleted unmatched orphan PNGs

- `image/Pasted image 20240511132845.png`
- `image/Pasted image 20240511135109.png`
- `image/Pasted image 20240511135940.png`
- `image/Pasted image 20240511155338.png`
- `image/Pasted image 20240511160805.png`
- `image/Pasted image 20240514091808.png`
- `image/Pasted image 20240514091859.png`
- `image/Pasted image 20240514092000.png`
- `image/Pasted image 20240514092134.png`
- `image/Pasted image 20240514092225.png`
- `image/Pasted image 20240516115214.png`
- `image/Pasted image 20240516115237.png`
- `image/Pasted image 20240516115656.png`
- `image/Pasted image 20240530112648.png`
- `image/Pasted image 20240610200559.png`
- `image/Pasted image 20240625161352.png`
- `image/Pasted image 20240625161535.png`
- `image/Pasted image 20240702190615.png`
- `image/Pasted image 20240816181906.png`
- `image/Pasted image 20240816181958.png`
- `image/Pasted image 20240816182019.png`
- `image/Pasted image 20240920131113.png`
- `image/Pasted image 20240920131318.png`
- `image/Pasted image 20240920131405.png`
- `image/Pasted image 20240920133558.png`
- `image/Pasted image 20240930155648.png`
- `image/Pasted image 20241007170948.png`
- `image/Pasted image 20241007171034.png`
- `image/Pasted image 20241007171129.png`
- `image/Pasted image 20241007171330.png`
- `image/Pasted image 20241015145352.png`
- `image/Pasted image 20241031102954.png`
- `image/Pasted image 20241104170231.png`
- `image/Pasted image 20241108112632.png`
- `image/Pasted image 20241108154405.png`
- `image/Pasted image 20241122150707.png`
- `image/Pasted image 20241122151046.png`
- `image/Pasted image 20241122151224.png`
- `image/Pasted image 20241122152428.png`
- `image/Pasted image 20241122152722.png`
- `image/Pasted image 20241122152920.png`
- `image/Pasted image 20241122153100.png`
- `image/Pasted image 20241122153139.png`
- `image/Pasted image 20241122154418.png`
- `image/Pasted image 20241122154958.png`
- `image/Pasted image 20241122155237.png`
- `image/Pasted image 20241122155311.png`
- `image/Pasted image 20241122160906.png`
- `image/Pasted image 20241122160927.png`
- `image/Pasted image 20241124154902.png`
- `image/Pasted image 20241125105456.png`
- `image/Pasted image 20241127174021.png`
- `image/Pasted image 20241129153733.png`
- `image/Pasted image 20241129153752.png`
- `image/Pasted image 20241129153804.png`
- `image/Pasted image 20241129153815.png`
- `image/Pasted image 20241129153943.png`
- `image/Pasted image 20241202095823.png`
- `image/Pasted image 20241207141445.png`
- `image/Pasted image 20241211133011.png`
- `image/Pasted image 20241211133659.png`
- `image/Pasted image 20241211155550.png`
- `image/Pasted image 20241213154815.png`
- `image/Pasted image 20241214132805.png`
- `image/Pasted image 20241215151624.png`
- `image/Pasted image 20241217152556.png`
- `image/Pasted image 20241217153013.png`
- `image/Pasted image 20241217153127.png`
- `image/Pasted image 20250912121248.png`
- `image/Pasted image 20250912121429.png`
- `image/Pasted image 20250919125428.png`
- `image/Pasted image 20250919183039.png`
- `image/Pasted image 20250919183131.png`

## Remaining unembedded PNGs
