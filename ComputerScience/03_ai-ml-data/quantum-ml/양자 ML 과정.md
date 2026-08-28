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
status: draft
tags:
- type/lecture
- quantum-computing
- machine-learning
title: 양자 ML 과정
type: lecture
updated: '2026-08-19'
---

graph:: [지식그래프 허브](<../../00_graph-interfaces/지식그래프 허브.md>)
domain:: [AI ML 데이터 인터페이스](<../AI ML 데이터 인터페이스.md>)
module:: [양자 ML 인터페이스](<../../00_graph-interfaces/courses/양자 ML 인터페이스.md>)
bridge:: [AI 구현 브리지](<../../00_graph-interfaces/bridges/AI 구현 브리지.md>), [수학 알고리즘 브리지](<../../00_graph-interfaces/bridges/수학 알고리즘 브리지.md>)
related:: [머신러닝 인터페이스](<../../00_graph-interfaces/courses/머신러닝 인터페이스.md>), [머신러닝](<../machine-learning/머신러닝 핵심 수학 개념.md>), [최적화 수학](<../../02_math-theory/optimization-math/1. Matrix/1. Matrix.md>)

> [!info] 강의 정리문서는 인덱스에서
> 원본 강의 PDF를 근거로 재작성한 정리문서 10편은 [00. 양자 ML 인덱스](<00. 양자 ML 인덱스.md>) 에 모여 있다.
> 이 노트는 과정 전체의 **진입점**으로 남는다.

방학 동안 진행하는 머신러닝 기반 양자컴퓨팅 과정의 수업 노트 진입점입니다.

## 수업 자료

- 양자 역학의 깊은 이론은 이 과정에서 깊게 다루지 않습니다.
- Day27은 `output/*.npy`, `*.pkl`, `*.csv`를 공유하는 순서형 실습입니다. 환경 설정부터 성능 보고서까지 순서대로 실행하세요.
- Day34는 `create_qnn`과 `torch_connector` 지원 모듈을 선행하는 순서형 QNN 실습입니다.
- Netflix CSV는 `.omo/drive-sync/raw/Day39/Project/`의 raw-only 데이터셋이며 Capstone 실습의 사전 조건입니다.

## 구현된 커리큘럼

### 01.quantum-foundations/01.why-quantum-and-qml

- [표현력의 한계](<01.quantum-foundations/01.why-quantum-and-qml/day01_02_expressive_power_limit_lecture.pdf>)
- [Quantum Computing이 필요한가](<01.quantum-foundations/01.why-quantum-and-qml/day02_01_why_quantum_computing_lecture.pdf>)
- [supplement_02_xor_expressive_power_classical_baselines_lab](<01.quantum-foundations/01.why-quantum-and-qml/supplement_02_xor_expressive_power_classical_baselines_lab.ipynb>)
- [week01_04_qml_quantum_role_numpy_feature_map_lab](<01.quantum-foundations/01.why-quantum-and-qml/week01_04_qml_quantum_role_numpy_feature_map_lab.ipynb>)

### 01.quantum-foundations/02.bits-qubits-and-state

- [Bit와 Qubit](<01.quantum-foundations/02.bits-qubits-and-state/day02_02_bit_and_qubit_lecture.pdf>)
- [day06_03_02_01_hadamard_counts_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_01_hadamard_counts_lab.ipynb>)
- [day06_03_02_02_hadamard_probability_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_02_hadamard_probability_lab.ipynb>)
- [day06_03_02_03_probability_vector_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_03_probability_vector_lab.ipynb>)
- [day06_03_02_04_zero_state_probability_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_04_zero_state_probability_lab.ipynb>)
- [day06_03_02_05_hadamard_state_probability_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_05_hadamard_state_probability_lab.ipynb>)
- [day06_03_02_06_x_state_probability_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_03_02_06_x_state_probability_lab.ipynb>)
- [day06_derived_03_02_01_to_03_02_06_measurement_probability_vectors_lab](<01.quantum-foundations/02.bits-qubits-and-state/day06_derived_03_02_01_to_03_02_06_measurement_probability_vectors_lab.ipynb>)

### 01.quantum-foundations/03.gates-measurement-and-entanglement

- [Hadamard Gate 개념 이해](<01.quantum-foundations/03.gates-measurement-and-entanglement/day05_03_hadamard_gate_concepts_lecture.pdf>)
- [상태변화 분석](<01.quantum-foundations/03.gates-measurement-and-entanglement/day05_04_quantum_state_transition_analysis_lecture.pdf>)
- [day06_02_01_hadamard_10_shots_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_02_01_hadamard_10_shots_lab.ipynb>)
- [day06_02_02_hadamard_100_shots_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_02_02_hadamard_100_shots_lab.ipynb>)
- [day06_02_03_hadamard_1000_shots_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_02_03_hadamard_1000_shots_lab.ipynb>)
- [day06_03_01_01_two_qubit_basis_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_03_01_01_two_qubit_basis_measurement_lab.ipynb>)
- [day06_03_01_02_single_hadamard_two_qubit_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_03_01_02_single_hadamard_two_qubit_lab.ipynb>)
- [day06_03_01_03_two_hadamard_two_qubit_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_03_01_03_two_hadamard_two_qubit_lab.ipynb>)
- [day06_04_01_zero_state_histogram_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_04_01_zero_state_histogram_lab.ipynb>)
- [day06_04_02_hadamard_histogram_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_04_02_hadamard_histogram_lab.ipynb>)
- [day06_04_03_x_gate_histogram_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_04_03_x_gate_histogram_lab.ipynb>)
- [day06_04_04_hx_gate_histogram_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_04_04_hx_gate_histogram_lab.ipynb>)
- [day06_derived_02_01_to_02_03_hadamard_shot_convergence_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_derived_02_01_to_02_03_hadamard_shot_convergence_lab.ipynb>)
- [day06_derived_03_01_01_to_03_01_03_two_qubit_hadamard_comparison_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_derived_03_01_01_to_03_01_03_two_qubit_hadamard_comparison_lab.ipynb>)
- [day06_derived_04_01_to_04_04_single_qubit_gate_effects_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day06_derived_04_01_to_04_04_single_qubit_gate_effects_lab.ipynb>)
- [day07_02_01_bell_state_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day07_02_01_bell_state_measurement_lab.ipynb>)
- [Quantum Gate 개념](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_01_quantum_gate_concepts_lecture.pdf>)
- [day08_03_01_cnot_basis_state_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_03_01_cnot_basis_state_lab.ipynb>)
- [day08_03_02_hadamard_superposition_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_03_02_hadamard_superposition_lab.ipynb>)
- [day08_03_03_bell_state_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_03_03_bell_state_lab.ipynb>)
- [day08_04_01_bell_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_01_bell_measurement_lab.ipynb>)
- [day08_04_02_01_bell_circuit_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_02_01_bell_circuit_lab.ipynb>)
- [day08_04_02_02_cnot_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_02_02_cnot_measurement_lab.ipynb>)
- [day08_04_02_03_hadamard_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_02_03_hadamard_measurement_lab.ipynb>)
- [day08_04_03_01_classical_random_bits_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_03_01_classical_random_bits_lab.ipynb>)
- [day08_04_03_02_bell_histogram_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day08_04_03_02_bell_histogram_lab.ipynb>)
- [day09_02_10_xx_identity_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_02_10_xx_identity_measurement_lab.ipynb>)
- [day09_02_11_hh_identity_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_02_11_hh_identity_measurement_lab.ipynb>)
- [day09_02_12_hzh_phase_flip_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_02_12_hzh_phase_flip_measurement_lab.ipynb>)
- [day09_x_02_01_x_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_x_02_01_x_gate_measurement_lab.ipynb>)
- [day09_y_02_05_y_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_y_02_05_y_gate_measurement_lab.ipynb>)
- [day09_y_02_06_xy_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_y_02_06_xy_gate_measurement_lab.ipynb>)
- [day09_y_02_07_x_y_gate_comparison_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_y_02_07_x_y_gate_comparison_lab.ipynb>)
- [day09_z_02_02_z_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_z_02_02_z_gate_measurement_lab.ipynb>)
- [day09_z_02_03_xz_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_z_02_03_xz_gate_measurement_lab.ipynb>)
- [day09_z_02_04_hzh_gate_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day09_z_02_04_hzh_gate_measurement_lab.ipynb>)
- [day10_02_01_hx_xh_order_measurement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day10_02_01_hx_xh_order_measurement_lab.ipynb>)
- [day10_02_02_multi_qubit_rotation_entanglement_lab](<01.quantum-foundations/03.gates-measurement-and-entanglement/day10_02_02_multi_qubit_rotation_entanglement_lab.ipynb>)

### 02.circuits-and-encoding/01.quantum-circuits-and-qml

- [Quantum Circuit 이란 ?](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day10_01_quantum_circuit_lecture.pdf>)
- [day11_01_01_feature_map_circuit_lab](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day11_01_01_feature_map_circuit_lab.ipynb>)
- [day11_03_01_hadamard_measurement_lab](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day11_03_01_hadamard_measurement_lab.ipynb>)
- [day11_03_02_superposition_rotation_entanglement_lab](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day11_03_02_superposition_rotation_entanglement_lab.ipynb>)
- [day11_03_03_composite_gate_measurement_lab](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day11_03_03_composite_gate_measurement_lab.ipynb>)
- [Quantum Circuit과 QML](<02.circuits-and-encoding/01.quantum-circuits-and-qml/day11_04_quantum_circuit_and_qml_lecture.pdf>)

### 02.circuits-and-encoding/02.feature-encoding

- [day01_03_02_synthetic_binary_classification_dataset_lab](<02.circuits-and-encoding/02.feature-encoding/day01_03_02_synthetic_binary_classification_dataset_lab.ipynb>)
- [day01_03_03_pca_projection_and_explained_variance_lab](<02.circuits-and-encoding/02.feature-encoding/day01_03_03_pca_projection_and_explained_variance_lab.ipynb>)
- [Quantum Feature Space](<02.circuits-and-encoding/02.feature-encoding/day02_03_quantum_feature_space_lecture.pdf>)
- [QML에서 Quantum의 역할](<02.circuits-and-encoding/02.feature-encoding/day02_04_quantum_role_in_qml_lecture.pdf>)
- [day09_04_01_feature_encoded_entangling_circuit_lab](<02.circuits-and-encoding/02.feature-encoding/day09_04_01_feature_encoded_entangling_circuit_lab.ipynb>)
- [day09_04_02_feature_encoded_no_cx_circuit_lab](<02.circuits-and-encoding/02.feature-encoding/day09_04_02_feature_encoded_no_cx_circuit_lab.ipynb>)
- [day09_04_03_feature_encoded_cx_circuit_lab](<02.circuits-and-encoding/02.feature-encoding/day09_04_03_feature_encoded_cx_circuit_lab.ipynb>)
- [day09_zz_04_04_feature_map_reps1_lab](<02.circuits-and-encoding/02.feature-encoding/day09_zz_04_04_feature_map_reps1_lab.ipynb>)
- [day09_zz_04_05_feature_map_dim3_lab](<02.circuits-and-encoding/02.feature-encoding/day09_zz_04_05_feature_map_dim3_lab.ipynb>)
- [day09_zz_04_06_feature_map_reps3_lab](<02.circuits-and-encoding/02.feature-encoding/day09_zz_04_06_feature_map_reps3_lab.ipynb>)
- [day12_01_02_customer_feature_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day12_01_02_customer_feature_encoding_lab.ipynb>)
- [day12_02_01_parameterized_feature_map_lab](<02.circuits-and-encoding/02.feature-encoding/day12_02_01_parameterized_feature_map_lab.ipynb>)
- [day12_02_full_quantum_feature_space_lab](<02.circuits-and-encoding/02.feature-encoding/day12_02_full_quantum_feature_space_lab.ipynb>)
- [day14_01_01_single_feature_angle_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day14_01_01_single_feature_angle_encoding_lab.ipynb>)
- [day14_01_02_two_feature_angle_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day14_01_02_two_feature_angle_encoding_lab.ipynb>)
- [day14_01_03_multi_feature_angle_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day14_01_03_multi_feature_angle_encoding_lab.ipynb>)
- [day14_01_04_basis_angle_hybrid_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day14_01_04_basis_angle_hybrid_encoding_lab.ipynb>)
- [day14_01_05_customer_hybrid_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day14_01_05_customer_hybrid_encoding_lab.ipynb>)
- [day15_02_01_amplitude_encoding_workflow_lab](<02.circuits-and-encoding/02.feature-encoding/day15_02_01_amplitude_encoding_workflow_lab.ipynb>)
- [day15_02_02_amplitude_encoding_simulation_lab](<02.circuits-and-encoding/02.feature-encoding/day15_02_02_amplitude_encoding_simulation_lab.ipynb>)
- [day15_02_03_amplitude_encoding_function_lab](<02.circuits-and-encoding/02.feature-encoding/day15_02_03_amplitude_encoding_function_lab.ipynb>)
- [day15_03_01_encoding_comparison_lab](<02.circuits-and-encoding/02.feature-encoding/day15_03_01_encoding_comparison_lab.ipynb>)
- [day16_02_01_single_pauli_feature_maps_lab](<02.circuits-and-encoding/02.feature-encoding/day16_02_01_single_pauli_feature_maps_lab.ipynb>)
- [day16_02_02_multi_pauli_feature_maps_lab](<02.circuits-and-encoding/02.feature-encoding/day16_02_02_multi_pauli_feature_maps_lab.ipynb>)
- [day16_02_03_combined_pauli_feature_maps_lab](<02.circuits-and-encoding/02.feature-encoding/day16_02_03_combined_pauli_feature_maps_lab.ipynb>)
- [day16_02_04_pauli_feature_map_iris_lab](<02.circuits-and-encoding/02.feature-encoding/day16_02_04_pauli_feature_map_iris_lab.ipynb>)
- [day16_03_01_sensor_phase_normalization_lab](<02.circuits-and-encoding/02.feature-encoding/day16_03_01_sensor_phase_normalization_lab.ipynb>)
- [day16_03_02_sensor_phase_encoding_circuit_lab](<02.circuits-and-encoding/02.feature-encoding/day16_03_02_sensor_phase_encoding_circuit_lab.ipynb>)
- [day16_03_03_sensor_phase_statevector_lab](<02.circuits-and-encoding/02.feature-encoding/day16_03_03_sensor_phase_statevector_lab.ipynb>)
- [day16_03_04_phase_interference_measurement_lab](<02.circuits-and-encoding/02.feature-encoding/day16_03_04_phase_interference_measurement_lab.ipynb>)
- [day18_04_01_frequency_encoding_lab](<02.circuits-and-encoding/02.feature-encoding/day18_04_01_frequency_encoding_lab.ipynb>)
- [day18_04_02_frequency_expectation_curve_lab](<02.circuits-and-encoding/02.feature-encoding/day18_04_02_frequency_expectation_curve_lab.ipynb>)
- [day18_04_03_time_evolution_statevector_lab](<02.circuits-and-encoding/02.feature-encoding/day18_04_03_time_evolution_statevector_lab.ipynb>)
- [day18_04_04_measurement_probability_oscillation_lab](<02.circuits-and-encoding/02.feature-encoding/day18_04_04_measurement_probability_oscillation_lab.ipynb>)

### 02.circuits-and-encoding/03.ansatz-and-parameterized-circuits

- [day11_04_01_qml_feature_map_ansatz_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day11_04_01_qml_feature_map_ansatz_lab.ipynb>)
- [day13_02_01_feature_map_ansatz_measurement_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day13_02_01_feature_map_ansatz_measurement_lab.ipynb>)
- [day13_04_01_mini_qml_pipeline_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day13_04_01_mini_qml_pipeline_lab.ipynb>)
- [day14_02_01_qml_end_to_end_workflow_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day14_02_01_qml_end_to_end_workflow_lab.ipynb>)
- [day17_01_01_parameter_sweep_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day17_01_01_parameter_sweep_lab.ipynb>)
- [day17_01_02_parameter_binding_state_analysis_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day17_01_02_parameter_binding_state_analysis_lab.ipynb>)
- [day17_02_01_real_amplitudes_ansatz_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day17_02_01_real_amplitudes_ansatz_lab.ipynb>)
- [day18_03_01_pennylane_strongly_entangling_classifier_training_lab](<02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/day18_03_01_pennylane_strongly_entangling_classifier_training_lab.ipynb>)

### 03.variational-learning-and-kernels/01.loss-and-optimization

- [day01_03_01_quadratic_loss_surface_visualization_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day01_03_01_quadratic_loss_surface_visualization_lab.ipynb>)
- [day19_01_01_quantum_loss_parameter_sweep_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day19_01_01_quantum_loss_parameter_sweep_lab.ipynb>)
- [day19_02_01_loss_function_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day19_02_01_loss_function_lab.ipynb>)
- [day20_02_02_manual_learning_cycle_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day20_02_02_manual_learning_cycle_lab.ipynb>)
- [day20_02_03_loss_landscape_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day20_02_03_loss_landscape_lab.ipynb>)
- [day20_02_04_parameter_sweep_prediction_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day20_02_04_parameter_sweep_prediction_lab.ipynb>)
- [day20_02_05_target_loss_curve_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day20_02_05_target_loss_curve_lab.ipynb>)
- [day21_03_01_common_quantum_objective_function_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_03_01_common_quantum_objective_function_lab.ipynb>)
- [day21_03_02_cobyla_optimizer_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_03_02_cobyla_optimizer_lab.ipynb>)
- [day21_03_03_spsa_optimizer_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_03_03_spsa_optimizer_lab.ipynb>)
- [day21_03_04_quantum_learning_visualization_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_03_04_quantum_learning_visualization_lab.ipynb>)
- [day21_04_01_before_after_learning_comparison_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_04_01_before_after_learning_comparison_lab.ipynb>)
- [day21_04_02_optimizer_comparison_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_04_02_optimizer_comparison_lab.ipynb>)
- [day21_04_03_learning_result_parameter_persistence_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day21_04_03_learning_result_parameter_persistence_lab.ipynb>)
- [day37_03_01_parameterized_quantum_circuit_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_01_parameterized_quantum_circuit_lab.ipynb>)
- [day37_03_02_parameter_binding_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_02_parameter_binding_lab.ipynb>)
- [day37_03_03_objective_function_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_03_objective_function_lab.ipynb>)
- [day37_03_04_classical_optimizer_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_04_classical_optimizer_lab.ipynb>)
- [day37_03_05_quantum_circuit_objective_function_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_05_quantum_circuit_objective_function_lab.ipynb>)
- [day37_03_06_cost_parameter_history_visualization_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_06_cost_parameter_history_visualization_lab.ipynb>)
- [day37_03_07_initial_parameter_comparison_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_07_initial_parameter_comparison_lab.ipynb>)
- [day37_03_08_optimizer_comparison_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_08_optimizer_comparison_lab.ipynb>)
- [day37_03_09_toy_trigonometric_objective_landscape_optimization_lab](<03.variational-learning-and-kernels/01.loss-and-optimization/day37_03_09_toy_trigonometric_objective_landscape_optimization_lab.ipynb>)

### 03.variational-learning-and-kernels/02.quantum-kernels

- [day15_01_01_zz_feature_map_quantum_kernel_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day15_01_01_zz_feature_map_quantum_kernel_lab.ipynb>)
- [day22_01_01_classical_to_quantum_kernel_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day22_01_01_classical_to_quantum_kernel_lab.ipynb>)
- [day23_03_01_fidelity_quantum_kernel_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day23_03_01_fidelity_quantum_kernel_lab.ipynb>)
- [day23_03_02_pauli_feature_map_quantum_kernel_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day23_03_02_pauli_feature_map_quantum_kernel_lab.ipynb>)
- [day23_04_01_example_kernel_matrix_structure_and_psd_validation_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day23_04_01_example_kernel_matrix_structure_and_psd_validation_lab.ipynb>)
- [day23_04_02_example_kernel_matrix_heatmap_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day23_04_02_example_kernel_matrix_heatmap_lab.ipynb>)
- [day23_04_03_quantum_kernel_quality_evaluation_lab](<03.variational-learning-and-kernels/02.quantum-kernels/day23_04_03_quantum_kernel_quality_evaluation_lab.ipynb>)

### 04.quantum-kernel-classification/01.iris-qsvm

- [day01_01_01_iris_environment_eda_logistic_knn_baselines_lab](<04.quantum-kernel-classification/01.iris-qsvm/day01_01_01_iris_environment_eda_logistic_knn_baselines_lab.ipynb>)
- [day01_01_02_iris_logistic_regression_and_visualization_lab](<04.quantum-kernel-classification/01.iris-qsvm/day01_01_02_iris_logistic_regression_and_visualization_lab.ipynb>)
- [day24_01_01_classical_svm_baseline_lab](<04.quantum-kernel-classification/01.iris-qsvm/day24_01_01_classical_svm_baseline_lab.ipynb>)
- [day24_02_01_qsvc_iris_classifier_lab](<04.quantum-kernel-classification/01.iris-qsvm/day24_02_01_qsvc_iris_classifier_lab.ipynb>)
- [day25_03_01_qsvm_training_data_preparation_lab](<04.quantum-kernel-classification/01.iris-qsvm/day25_03_01_qsvm_training_data_preparation_lab.ipynb>)
- [day25_03_02_qsvc_model_construction_lab](<04.quantum-kernel-classification/01.iris-qsvm/day25_03_02_qsvc_model_construction_lab.ipynb>)
- [day25_03_03_qsvm_training_lab](<04.quantum-kernel-classification/01.iris-qsvm/day25_03_03_qsvm_training_lab.ipynb>)
- [day25_03_04_qsvm_prediction_lab](<04.quantum-kernel-classification/01.iris-qsvm/day25_03_04_qsvm_prediction_lab.ipynb>)
- [day26_01_01_classification_metrics_confusion_matrix_lab](<04.quantum-kernel-classification/01.iris-qsvm/day26_01_01_classification_metrics_confusion_matrix_lab.ipynb>)
- [day26_04_01_qsvm_performance_evaluation_lab](<04.quantum-kernel-classification/01.iris-qsvm/day26_04_01_qsvm_performance_evaluation_lab.ipynb>)
- [day27_02_01_qsvm_environment_setup_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_01_qsvm_environment_setup_lab.ipynb>)
- [day27_02_02_qsvm_data_preparation_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_02_qsvm_data_preparation_lab.ipynb>)
- [day27_02_03_qsvm_training_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_03_qsvm_training_lab.ipynb>)
- [day27_02_04_qsvm_prediction_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_04_qsvm_prediction_lab.ipynb>)
- [day27_02_05_qsvm_accuracy_analysis_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_05_qsvm_accuracy_analysis_lab.ipynb>)
- [day27_02_06_qsvm_precision_recall_f1_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_06_qsvm_precision_recall_f1_lab.ipynb>)
- [day27_02_07_qsvm_classification_report_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_07_qsvm_classification_report_lab.ipynb>)
- [day27_02_08_qsvm_confusion_matrix_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_08_qsvm_confusion_matrix_lab.ipynb>)
- [day27_02_09_qsvm_roc_auc_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_09_qsvm_roc_auc_lab.ipynb>)
- [day27_02_10_qsvm_performance_report_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_02_10_qsvm_performance_report_lab.ipynb>)
- [day27_04_01_classical_svm_performance_baseline_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_04_01_classical_svm_performance_baseline_lab.ipynb>)
- [day27_04_02_qsvm_performance_analysis_improvement_lab](<04.quantum-kernel-classification/01.iris-qsvm/day27_04_02_qsvm_performance_analysis_improvement_lab.ipynb>)

### 05.quantum-neural-networks/01.classical-neural-network-baseline

- [day28_01_01_neural_network_structure_lab](<05.quantum-neural-networks/01.classical-neural-network-baseline/day28_01_01_neural_network_structure_lab.ipynb>)
- [day33_01_01_pytorch_basic_model_lab](<05.quantum-neural-networks/01.classical-neural-network-baseline/day33_01_01_pytorch_basic_model_lab.ipynb>)

### 05.quantum-neural-networks/02.estimator-qnn

- [day31_02_01_qnn_components_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_02_01_qnn_components_lab.ipynb>)
- [day31_03_01_parameterized_quantum_circuit_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_03_01_parameterized_quantum_circuit_lab.ipynb>)
- [day31_03_02_estimator_qnn_preparation_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_03_02_estimator_qnn_preparation_lab.ipynb>)
- [day31_03_03_estimator_qnn_construction_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_03_03_estimator_qnn_construction_lab.ipynb>)
- [day31_03_04_estimator_qnn_forward_pass_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_03_04_estimator_qnn_forward_pass_lab.ipynb>)
- [day31_03_05_estimator_qnn_forward_analysis_lab](<05.quantum-neural-networks/02.estimator-qnn/day31_03_05_estimator_qnn_forward_analysis_lab.ipynb>)
- [day34_03_02_estimator_qnn_creation_lab](<05.quantum-neural-networks/02.estimator-qnn/day34_03_02_estimator_qnn_creation_lab.ipynb>)
- [day34_create_qnn_support_module_lab](<05.quantum-neural-networks/02.estimator-qnn/day34_create_qnn_support_module_lab.ipynb>)

### 05.quantum-neural-networks/03.torchconnector-hybrid-qnn

- [day34_02_01_torchconnector_pytorch_integration_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_02_01_torchconnector_pytorch_integration_lab.ipynb>)
- [day34_03_01_torchconnector_environment_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_01_torchconnector_environment_lab.ipynb>)
- [day34_03_03_torchconnector_application_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_03_torchconnector_application_lab.ipynb>)
- [day34_03_04_quantum_weight_parameter_analysis_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_04_quantum_weight_parameter_analysis_lab.ipynb>)
- [day34_03_05_qnn_forward_batch_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_05_qnn_forward_batch_lab.ipynb>)
- [day34_03_06_qnn_input_change_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_06_qnn_input_change_lab.ipynb>)
- [day34_03_07_qnn_weight_change_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_07_qnn_weight_change_lab.ipynb>)
- [day34_03_08_qnn_gradient_calculation_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_08_qnn_gradient_calculation_lab.ipynb>)
- [day34_03_09_qnn_batch_forward_training_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_09_qnn_batch_forward_training_lab.ipynb>)
- [day34_03_10_qnn_end_to_end_validation_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_03_10_qnn_end_to_end_validation_lab.ipynb>)
- [day34_torch_connector_support_module_lab](<05.quantum-neural-networks/03.torchconnector-hybrid-qnn/day34_torch_connector_support_module_lab.ipynb>)

### 06.qaoa-and-combinatorial-optimization/01.tsp-classical-baseline

- [day35_01_01_tsp_brute_force_baseline_lab](<06.qaoa-and-combinatorial-optimization/01.tsp-classical-baseline/day35_01_01_tsp_brute_force_baseline_lab.ipynb>)

### 06.qaoa-and-combinatorial-optimization/02.qaoa-objectives-and-optimizers

- [day38_01_02_qaoa_import_object_setup_lab](<06.qaoa-and-combinatorial-optimization/02.qaoa-objectives-and-optimizers/day38_01_02_qaoa_import_object_setup_lab.ipynb>)
- [day38_01_04_qaoa_classical_comparison_lab](<06.qaoa-and-combinatorial-optimization/02.qaoa-objectives-and-optimizers/day38_01_04_qaoa_classical_comparison_lab.ipynb>)

### 06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution

- [day38_01_01_qaoa_environment_check_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day38_01_01_qaoa_environment_check_lab.ipynb>)
- [day38_01_03_first_qaoa_execution_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day38_01_03_first_qaoa_execution_lab.ipynb>)
- [day39_02_01_cost_parameter_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_01_cost_parameter_lab.ipynb>)
- [day39_02_02_cost_layer_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_02_cost_layer_lab.ipynb>)
- [day39_02_03_cost_circuit_output_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_03_cost_circuit_output_lab.ipynb>)
- [day39_02_04_gamma_parameter_effect_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_04_gamma_parameter_effect_lab.ipynb>)
- [day39_02_05_mixer_layer_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_05_mixer_layer_lab.ipynb>)
- [day39_02_06_parameterized_qaoa_circuit_lab](<06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/day39_02_06_parameterized_qaoa_circuit_lab.ipynb>)

### 07.capstone/01.netflix-qml-project

- [day39_project_01_01_netflix_dataset_quality_and_leakage_audit_lab](<07.capstone/01.netflix-qml-project/day39_project_01_01_netflix_dataset_quality_and_leakage_audit_lab.ipynb>)
- [day39_project_01_02_netflix_classical_feature_preparation_for_qml_lab](<07.capstone/01.netflix-qml-project/day39_project_01_02_netflix_classical_feature_preparation_for_qml_lab.ipynb>)
