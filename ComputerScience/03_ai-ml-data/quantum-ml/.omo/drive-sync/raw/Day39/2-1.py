# ============================================================
# Part 2. Cost Hamiltonian 구현
# Lab 1. Cost Parameter 생성
# ============================================================

from qiskit.circuit import Parameter


# ------------------------------------------------------------
# Step 1. Cost Parameter γ 생성
# ------------------------------------------------------------

gamma = Parameter("γ")


# ------------------------------------------------------------
# Step 2. 생성된 Parameter 출력
# ------------------------------------------------------------

print("=== Cost Parameter ===")
print(gamma)


# ------------------------------------------------------------
# Step 3. Parameter Type 확인
# ------------------------------------------------------------

print("\n=== Parameter Type ===")
print(type(gamma))


# ------------------------------------------------------------
# Step 4. Parameter 이름 확인
# ------------------------------------------------------------

print("\n=== Parameter Name ===")
print(gamma.name)


# ------------------------------------------------------------
# Step 5. 일반 숫자와 Parameter 비교
# ------------------------------------------------------------

gamma_number = 0.5

print("\n=== Number vs Parameter ===")

print("Fixed Number :", gamma_number)
print("Number Type  :", type(gamma_number))

print("Parameter    :", gamma)
print("Parameter Type:", type(gamma))


# ------------------------------------------------------------
# Step 6. Parameter Expression 생성
# ------------------------------------------------------------

theta = 2 * gamma

print("\n=== Parameter Expression ===")
print("gamma       =", gamma)
print("2 * gamma   =", theta)


# ------------------------------------------------------------
# Step 7. 최종 확인
# ------------------------------------------------------------

print("\n=== Lab 1 Result ===")
print("Cost Parameter :", gamma)
print("Rotation Angle :", theta)
print("Status         : Symbolic Parameter 생성 완료")