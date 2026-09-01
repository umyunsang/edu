"""
============================================================
Lab 1.
Smart Factory Sensor Data → Phase 변환
============================================================

실습 순서
---------------------------------------------
STEP 1. Import Library
STEP 2. Sensor Data 준비
STEP 3. Data 확인
STEP 4. Normalization
STEP 5. Phase Angle 계산
STEP 6. 결과 확인
STEP 7. 여러 Sample 비교
STEP 8. 결과 분석
============================================================
"""

import numpy as np
import pandas as pd

# ============================================================
# STEP 1.
# Sensor Data 준비
# ============================================================

print("=" * 80)
print("STEP 1. Smart Factory Sensor Data")
print("=" * 80)

sensor = {

    "Temperature":72,
    "Pressure":6.5,
    "Vibration":40

}

print(sensor)

# ============================================================
# STEP 2.
# Data 확인
# ============================================================

print()
print("=" * 80)
print("STEP 2. Raw Sensor Data")
print("=" * 80)

print(f"Temperature : {sensor['Temperature']} ℃")
print(f"Pressure    : {sensor['Pressure']} bar")
print(f"Vibration   : {sensor['Vibration']}")

# ============================================================
# STEP 3.
# Normalization
# ============================================================

print()
print("=" * 80)
print("STEP 3. Normalization")
print("=" * 80)

"""
Temperature
20 ~ 120 ℃

Pressure
1 ~ 10 bar

Vibration
0 ~ 100
"""

temp_norm = (sensor["Temperature"] - 20) / (120 - 20)

pressure_norm = (sensor["Pressure"] - 1) / (10 - 1)

vibration_norm = sensor["Vibration"] / 100

print("Normalized Value")

print(f"Temperature : {temp_norm:.4f}")

print(f"Pressure    : {pressure_norm:.4f}")

print(f"Vibration   : {vibration_norm:.4f}")

# ============================================================
# STEP 4.
# Phase Angle 계산
# ============================================================

print()
print("=" * 80)
print("STEP 4. Phase Angle")
print("=" * 80)


phase_temp = temp_norm * np.pi

phase_pressure = pressure_norm * np.pi

phase_vibration = vibration_norm * np.pi

print(f"Temperature Phase : {phase_temp:.4f} rad")

print(f"Pressure Phase    : {phase_pressure:.4f} rad")

print(f"Vibration Phase   : {phase_vibration:.4f} rad")

# ============================================================
# STEP 5.
# Degree 변환
# ============================================================

print()
print("=" * 80)
print("STEP 5. Degree")
print("=" * 80)

degree_temp = np.degrees(phase_temp)

degree_pressure = np.degrees(phase_pressure)

degree_vibration = np.degrees(phase_vibration)

print(f"Temperature : {degree_temp:.2f}°")

print(f"Pressure    : {degree_pressure:.2f}°")

print(f"Vibration   : {degree_vibration:.2f}°")

# ============================================================
# STEP 6.
# 결과 정리
# ============================================================

print()
print("=" * 80)
print("STEP 6. Summary Table")
print("=" * 80)

df = pd.DataFrame({

    "Feature":[

        "Temperature",
        "Pressure",
        "Vibration"

    ],

    "Raw Value":[

        sensor["Temperature"],
        sensor["Pressure"],
        sensor["Vibration"]

    ],

    "Normalized":[

        temp_norm,
        pressure_norm,
        vibration_norm

    ],

    "Phase(rad)":[

        phase_temp,
        phase_pressure,
        phase_vibration

    ],

    "Phase(deg)":[

        degree_temp,
        degree_pressure,
        degree_vibration

    ]

})

print(df)

# ============================================================
# STEP 7.
# 여러 Sample 비교
# ============================================================

print()
print("=" * 80)
print("STEP 7. Multiple Samples")
print("=" * 80)

samples = [

    {
        "Temperature":30,
        "Pressure":4,
        "Vibration":20
    },

    {
        "Temperature":70,
        "Pressure":6,
        "Vibration":40
    },

    {
        "Temperature":100,
        "Pressure":8,
        "Vibration":90
    }

]

rows = []

for sample in samples:

    t = (sample["Temperature"]-20)/(120-20)
    p = (sample["Pressure"]-1)/(10-1)
    v = sample["Vibration"]/100

    rows.append({

        "Temperature":sample["Temperature"],
        "Pressure":sample["Pressure"],
        "Vibration":sample["Vibration"],

        "T Phase(rad)":t*np.pi,
        "P Phase(rad)":p*np.pi,
        "V Phase(rad)":v*np.pi

    })

compare = pd.DataFrame(rows)

print(compare)


print("=" * 80)
print("Lab 1 Complete")
print("=" * 80)
