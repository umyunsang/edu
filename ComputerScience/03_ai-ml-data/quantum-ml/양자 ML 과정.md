---
aliases:
- 머신러닝 기반 양자컴퓨팅 과정
course: quantum-ml
created: '2026-06-22'
date: '2026-06-22'
kg_graph_size: 60
kg_layer_label: L4 support
kg_level: 4
kg_role: support
semester: summer
source: ''
status: seedling
tags:
- type/lecture
- quantum-computing
- machine-learning
title: 양자 ML 과정
type: lecture
updated: '2026-07-23'
---

graph:: [[ComputerScience/00_graph-interfaces/지식그래프 허브|지식그래프 허브]]
domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/양자 ML 인터페이스|양자 ML 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
related:: [[ComputerScience/00_graph-interfaces/courses/머신러닝 인터페이스|머신러닝 인터페이스]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|최적화 수학]]

# 양자 ML 과정

방학 동안 진행하는 머신러닝 기반 양자컴퓨팅 과정의 수업 노트 진입점입니다.

## 그래프 연결

- 분야: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
- 과목: [[ComputerScience/00_graph-interfaces/courses/양자 ML 인터페이스|양자 ML 인터페이스]]
- 브리지: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]]
- 브리지: [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]

## 수업 자료

- 강의 노트와 실습 파일은 커리큘럼 주제 폴더에 추가합니다.
- 실습 노트북은 `notebooks/practice/`처럼 깊은 하위 폴더를 만들지 않고 주제 폴더 바로 아래에 둡니다.
- 양자 역학의 깊은 이론은 이 과정에서 깊게 다루지 않습니다.

## 커리큘럼 구조

- `1.quantum-ml-overview/` — 양자 ML 개요
- `2.qml-structure/` — QML 구조 이해
- `1.quantum-ml-overview/qml-pipeline/` — QML 파이프라인
- `1.quantum-ml-overview/entanglement-and-cnot/` — Entanglement와 CNOT 실습
- `4.qml-models/` — QML 모델
- `5.qnn-and-hybrid/` — 양자 신경망(QNN) 및 Hybrid
- `6.performance-comparison-and-qaoa-basics/` — 성능 비교 및 QAOA 기초
- `7.qaoa-extension-and-industry-applications/` — QAOA 확장 및 산업 적용
- `8.mini-project/` — 미니 프로젝트

## 현재 자료 배치

- Hadamard Gate 개념 강의자료: `1.quantum-ml-overview/hadamard-gate/3. Hadamard Gate 개념 이해.pdf`
- Hadamard 측정 shots 실습: `1.quantum-ml-overview/hadamard-gate/2_hadamard_shots_practice.ipynb`
- 2-qubit Hadamard superposition 실습 3-1: `1.quantum-ml-overview/hadamard-gate/3_1_two_qubit_hadamard_superposition.ipynb`
- 상태변화 분석 강의자료: `1.quantum-ml-overview/state-change-analysis/4. 상태변화 분석.pdf`
- 측정 probability vector 실습 3-2: `1.quantum-ml-overview/state-change-analysis/3_2_measurement_probability_vectors.ipynb`
- single-qubit gate effects 실습 4: `1.quantum-ml-overview/state-change-analysis/4_single_qubit_gate_effects.ipynb`
- Quantum Computing 필요성 강의자료: `1.quantum-ml-overview/why-quantum-computing/1. 왜 Quantum Computing이 필요한가.pdf`
- Bit와 Qubit 강의자료: `1.quantum-ml-overview/bit-and-qubit/2. Bit와 Qubit.pdf`
- Quantum Feature Space 강의자료: `1.quantum-ml-overview/quantum-feature-space/3. Quantum Feature Space.pdf`
- QML에서 Quantum의 역할 강의자료: `1.quantum-ml-overview/quantum-role/week1_qml_quantum_role.pdf`
- QML에서 Quantum의 역할 실습: `1.quantum-ml-overview/quantum-role/4_qml_quantum_role_practice.ipynb`
- Iris 실습: `1.quantum-ml-overview/iris-classification/`
- 표현력의 한계 강의자료: `1.quantum-ml-overview/expressive-power-limit/lecture-materials/2. 표현력의 한계.pdf`
- 표현력의 한계 실습: `1.quantum-ml-overview/expressive-power-limit/2_xor_expressive_power_practice.ipynb`
- QML 파이프라인 실습 3-1: `1.quantum-ml-overview/qml-pipeline/3_1_loss_surface_visualization.ipynb`
- QML 파이프라인 실습 3-2: `1.quantum-ml-overview/qml-pipeline/3_2_binary_classification_dataset.ipynb`
- QML 파이프라인 실습 3-3: `1.quantum-ml-overview/qml-pipeline/3_3_pca_projection_visualization.ipynb`
- CNOT basis-state 실습 3-1: `1.quantum-ml-overview/entanglement-and-cnot/3_1_cnot_basis_state_practice.ipynb`
- Hadamard superposition 실습 3-2: `1.quantum-ml-overview/entanglement-and-cnot/3_2_hadamard_superposition_practice.ipynb`
- Bell state 실습 3-3: `1.quantum-ml-overview/entanglement-and-cnot/3_3_bell_state_practice.ipynb`
- Bell measurement 실습 4-1: `1.quantum-ml-overview/entanglement-and-cnot/4_1_bell_measurement_practice.ipynb`
- Bell circuit 실습 4-2-1: `1.quantum-ml-overview/entanglement-and-cnot/4_2_1_bell_circuit_practice.ipynb`
- CNOT measurement 실습 4-2-2: `1.quantum-ml-overview/entanglement-and-cnot/4_2_2_cnot_measurement_practice.ipynb`
- Hadamard measurement 실습 4-2-3: `1.quantum-ml-overview/entanglement-and-cnot/4_2_3_hadamard_measurement_practice.ipynb`
- classical random bits 실습 4-3-1: `1.quantum-ml-overview/entanglement-and-cnot/4_3_1_classical_random_bits_practice.ipynb`
- Bell histogram 실습 4-3-2: `1.quantum-ml-overview/entanglement-and-cnot/4_3_2_bell_histogram_practice.ipynb`
- Day9 X gate 측정 실습: `1.quantum-ml-overview/state-change-analysis/day9_x_2_1_x_gate_measurement_practice.ipynb`
- Day9 Z/H/Y gate 측정 실습: `1.quantum-ml-overview/state-change-analysis/`
- Day9 feature-encoded circuit 실습: `1.quantum-ml-overview/quantum-feature-space/day9_4_1_feature_encoded_entangling_circuit_practice.ipynb`
- Day9 ZZ feature map 실습: `1.quantum-ml-overview/quantum-feature-space/day9_zz_4_4_feature_map_reps1_practice.ipynb`
- Day10 H/X gate 순서 측정 실습: `1.quantum-ml-overview/state-change-analysis/day10_2_1_hx_xh_order_measurement_practice.ipynb`
- Day10 multi-qubit rotation/entanglement 실습: `1.quantum-ml-overview/state-change-analysis/day10_2_2_multi_qubit_rotation_entanglement_practice.ipynb`

- Quantum Gate 개념 강의자료: `1.quantum-ml-overview/quantum-gate/1. Quantum Gate 개념.pdf`
- Quantum Circuit 개요 강의자료: `1.quantum-ml-overview/quantum-circuit/1. Quantum Circuit 개요.pdf`
- Quantum Circuit과 QML 강의자료: `1.quantum-ml-overview/qml-circuit/4. Quantum Circuit과 QML.pdf`
- Day11 feature map circuit 실습: `1.quantum-ml-overview/qml-circuit/day11_1_1_feature_map_circuit_practice.ipynb`
- Day11 quantum circuit 측정 실습: `1.quantum-ml-overview/quantum-circuit/`
- Day11 QML feature map ansatz 실습: `1.quantum-ml-overview/qml-circuit/day11_4_1_qml_feature_map_ansatz_practice.ipynb`
- Day12 customer feature encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day12_1_2_customer_feature_encoding_practice.ipynb`
- Day12 parameterized feature map 실습: `1.quantum-ml-overview/quantum-feature-space/day12_2_1_parameterized_feature_map_practice.ipynb`
- Day12 Quantum Feature Space 종합 실습: `1.quantum-ml-overview/quantum-feature-space/day12_2_full_quantum_feature_space_lab.ipynb`
- Day13 feature map ansatz 측정 실습: `1.quantum-ml-overview/qml-circuit/day13_2_1_feature_map_ansatz_measurement_practice.ipynb`
- Day13 Mini QML Pipeline 실습: `1.quantum-ml-overview/qml-pipeline/day13_4_1_mini_qml_pipeline_practice.ipynb`
- Day14 single-feature angle encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day14_1_1_single_feature_angle_encoding_practice.ipynb`
- Day14 two-feature angle encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day14_1_2_two_feature_angle_encoding_practice.ipynb`
- Day14 multi-feature angle encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day14_1_3_multi_feature_angle_encoding_practice.ipynb`
- Day14 basis/angle/hybrid encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day14_1_4_basis_angle_hybrid_encoding_practice.ipynb`
- Day14 customer hybrid encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day14_1_5_customer_hybrid_encoding_practice.ipynb`
- Day14 QML end-to-end workflow 실습: `1.quantum-ml-overview/qml-pipeline/day14_2_1_qml_end_to_end_workflow_practice.ipynb`
- Day15 ZZ feature map와 quantum kernel 실습: `1.quantum-ml-overview/quantum-feature-space/day15_1_1_zz_feature_map_quantum_kernel_practice.ipynb`
- Day15 amplitude encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day15_2_1_amplitude_encoding_workflow_practice.ipynb`
- Day15 amplitude encoding 시뮬레이션 실습: `1.quantum-ml-overview/quantum-feature-space/day15_2_2_amplitude_encoding_simulation_practice.ipynb`
- Day15 amplitude encoding 함수화 실습: `1.quantum-ml-overview/quantum-feature-space/day15_2_3_amplitude_encoding_function_practice.ipynb`
- Day15 basis/angle/amplitude encoding 비교 실습: `1.quantum-ml-overview/quantum-feature-space/day15_3_1_encoding_comparison_practice.ipynb`
- Day6 Hadamard shots 실습: `1.quantum-ml-overview/hadamard-gate/day6_2_1_hadamard_10_shots_practice.ipynb`, `day6_2_2_hadamard_100_shots_practice.ipynb`, `day6_2_3_hadamard_1000_shots_practice.ipynb`
- Day6 two-qubit Hadamard 실습: `1.quantum-ml-overview/hadamard-gate/day6_3_1_1_two_qubit_basis_measurement_practice.ipynb`, `day6_3_1_2_single_hadamard_two_qubit_practice.ipynb`, `day6_3_1_3_two_hadamard_two_qubit_practice.ipynb`
- Day6 상태·확률 벡터 실습: `1.quantum-ml-overview/state-change-analysis/day6_3_2_1_hadamard_counts_practice.ipynb`부터 `day6_3_2_6_x_state_probability_practice.ipynb`까지
- Day6 상태 히스토그램 실습: `1.quantum-ml-overview/state-change-analysis/day6_4_1_zero_state_histogram_practice.ipynb`부터 `day6_4_4_hx_gate_histogram_practice.ipynb`까지
- Day7 Bell state 측정 실습: `1.quantum-ml-overview/entanglement-and-cnot/day7_2_1_bell_state_measurement_practice.ipynb`
- Day16 single/multi/combined Pauli feature map 실습: `1.quantum-ml-overview/quantum-feature-space/day16_2_1_single_pauli_feature_maps_practice.ipynb`부터 `day16_2_3_combined_pauli_feature_maps_practice.ipynb`까지
- Day16 Pauli feature map Iris 실습: `1.quantum-ml-overview/quantum-feature-space/day16_2_4_pauli_feature_map_iris_practice.ipynb`
- Day16 센서 phase encoding 실습: `1.quantum-ml-overview/quantum-feature-space/day16_3_1_sensor_phase_normalization_practice.ipynb`부터 `day16_3_4_phase_interference_measurement_practice.ipynb`까지
- Day17 parameter sweep·binding 실습: `1.quantum-ml-overview/qml-circuit/day17_1_1_parameter_sweep_practice.ipynb`, `day17_1_2_parameter_binding_state_analysis_practice.ipynb`
- Day17 RealAmplitudes ansatz 실습: `1.quantum-ml-overview/qml-circuit/day17_2_1_real_amplitudes_ansatz_practice.ipynb`
- Day18 Strongly Entangling Layers 실습: `1.quantum-ml-overview/qml-circuit/day18_3_1_strongly_entangling_layers_practice.ipynb`
- Day18 frequency encoding·expectation·time evolution·measurement 실습: `1.quantum-ml-overview/quantum-feature-space/day18_4_1_frequency_encoding_practice.ipynb`부터 `day18_4_4_measurement_probability_oscillation_practice.ipynb`까지
- Day19 quantum loss parameter sweep 실습: `1.quantum-ml-overview/qml-pipeline/day19_1_1_quantum_loss_parameter_sweep_practice.ipynb`
- Day19 loss function 실습: `1.quantum-ml-overview/qml-pipeline/day19_2_1_loss_function_practice.ipynb`
- Day20 manual learning cycle 실습: `1.quantum-ml-overview/qml-pipeline/day20_2_2_manual_learning_cycle_practice.ipynb`
- Day20 loss landscape·parameter sweep·target loss curve 실습: `1.quantum-ml-overview/qml-pipeline/day20_2_3_loss_landscape_practice.ipynb`부터 `day20_2_5_target_loss_curve_practice.ipynb`까지
- Day21 공통 objective function·COBYLA·SPSA·학습 시각화 실습: `1.quantum-ml-overview/qml-pipeline/day21_3_1_common_quantum_objective_function_practice.ipynb`부터 `day21_3_4_quantum_learning_visualization_practice.ipynb`까지
- Day21 학습 전후 비교·optimizer 비교·parameter 저장 실습: `1.quantum-ml-overview/qml-pipeline/day21_4_1_before_after_learning_comparison_practice.ipynb`부터 `day21_4_3_learning_result_parameter_persistence_practice.ipynb`까지
- Day22 Classical Kernel에서 Quantum Kernel으로 실습: `1.quantum-ml-overview/quantum-feature-space/day22_1_1_classical_to_quantum_kernel_practice.ipynb`
- Day23 Fidelity quantum kernel 실습: `1.quantum-ml-overview/quantum-feature-space/day23_3_1_fidelity_quantum_kernel_practice.ipynb`
- Day23 Pauli feature map quantum kernel 실습: `1.quantum-ml-overview/quantum-feature-space/day23_3_2_pauli_feature_map_quantum_kernel_practice.ipynb`
- Day23 quantum kernel matrix 검증 실습: `1.quantum-ml-overview/quantum-feature-space/day23_4_1_quantum_kernel_matrix_validation_practice.ipynb`
- Day23 quantum kernel heatmap 실습: `1.quantum-ml-overview/quantum-feature-space/day23_4_2_quantum_kernel_heatmap_practice.ipynb`
- Day23 quantum kernel 품질 평가 실습: `1.quantum-ml-overview/quantum-feature-space/day23_4_3_quantum_kernel_quality_evaluation_practice.ipynb`
- Day24 Classical SVM baseline 실습: `1.quantum-ml-overview/iris-classification/day24_1_1_classical_svm_baseline_practice.ipynb`
- Day24 Iris QSVC 분류 실습: `1.quantum-ml-overview/iris-classification/day24_2_1_qsvc_iris_classifier_practice.ipynb`
- Day25 QSVM 학습 데이터 준비 실습: `1.quantum-ml-overview/iris-classification/day25_3_1_qsvm_training_data_preparation_practice.ipynb`
- Day25 QSVC 모델 생성 실습: `1.quantum-ml-overview/iris-classification/day25_3_2_qsvc_model_construction_practice.ipynb`
- Day25 QSVM 학습 실습: `1.quantum-ml-overview/iris-classification/day25_3_3_qsvm_training_practice.ipynb`
- Day25 QSVM 예측 실습: `1.quantum-ml-overview/iris-classification/day25_3_4_qsvm_prediction_practice.ipynb`
- Day26 분류 평가 지표와 Confusion Matrix 실습: `1.quantum-ml-overview/iris-classification/day26_1_1_classification_metrics_confusion_matrix_practice.ipynb`
- Day26 QSVM 성능 평가 실습: `1.quantum-ml-overview/iris-classification/day26_4_1_qsvm_performance_evaluation_practice.ipynb`
