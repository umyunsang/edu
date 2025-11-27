
---
# REPORT

## 2장 연습 문제

![[Pasted image 20250329135531.png]]

|         |                       |
| ------- | :-------------------: |
| **과목명** |         최적화수학         |
| 학과      |         AI학과          |
| 학번      |        1705817        |
| 이름      |          엄윤상          |
| 제출일자    | 12월 7일 기준 오후 10:00 까지 |
| 담당교수    |        장재훈 교수님        |

---

## 문제 1. p156. 05 (a) - 방향 도함수

> **문제**: 주어진 방향에 대해 방향 도함수를 구하세요. (방향벡터의 크기 유지)
>
> $$f(x,y) = \frac{2x^2}{y}, \quad \mathbf{u} = (1,2)$$

### 풀이

**Step 1: 편미분 계산**

함수 $f(x,y) = \frac{2x^2}{y}$의 편미분을 구합니다.

$$\frac{\partial f}{\partial x} = \frac{4x}{y}$$

$$\frac{\partial f}{\partial y} = \frac{-2x^2}{y^2}$$

**Step 2: 그래디언트 벡터 구하기**

$$\nabla f(x,y) = \left(\frac{4x}{y}, \frac{-2x^2}{y^2}\right)$$

**Step 3: 방향 도함수 공식 적용**

방향벡터의 크기를 유지하므로 $\mathbf{u} = (1,2)$를 그대로 사용합니다.

방향 도함수는 다음과 같이 계산됩니다:

$$D_{\mathbf{u}}f(x,y) = \nabla f(x,y) \cdot \mathbf{u}$$

$$= \left(\frac{4x}{y}, \frac{-2x^2}{y^2}\right) \cdot (1,2)$$

$$= \frac{4x}{y} \cdot 1 + \frac{-2x^2}{y^2} \cdot 2$$

$$= \frac{4x}{y} - \frac{4x^2}{y^2}$$

$$= \frac{4xy - 4x^2}{y^2}$$

$$= \frac{4x(y - x)}{y^2}$$

### 답

$$\boxed{D_{\mathbf{u}}f(x,y) = \frac{4x(y - x)}{y^2}}$$

---

## 문제 2. p156. 05 (b) - 방향 도함수

> **문제**: 주어진 방향에 대해 방향 도함수를 구하세요. (방향벡터의 크기 유지)
>
> $$f(x,y,z) = x^2 e^{yz}, \quad \mathbf{u} = (1,-1,3)$$

### 풀이

**Step 1: 편미분 계산**

함수 $f(x,y,z) = x^2 e^{yz}$의 편미분을 구합니다.

$$\frac{\partial f}{\partial x} = 2x e^{yz}$$

$$\frac{\partial f}{\partial y} = x^2 z e^{yz}$$

$$\frac{\partial f}{\partial z} = x^2 y e^{yz}$$

**Step 2: 그래디언트 벡터 구하기**

$$\nabla f(x,y,z) = \left(2x e^{yz}, x^2 z e^{yz}, x^2 y e^{yz}\right)$$

**Step 3: 방향 도함수 공식 적용**

방향벡터의 크기를 유지하므로 $\mathbf{u} = (1,-1,3)$를 그대로 사용합니다.

$$D_{\mathbf{u}}f(x,y,z) = \nabla f(x,y,z) \cdot \mathbf{u}$$

$$= \left(2x e^{yz}, x^2 z e^{yz}, x^2 y e^{yz}\right) \cdot (1,-1,3)$$

$$= 2x e^{yz} \cdot 1 + x^2 z e^{yz} \cdot (-1) + x^2 y e^{yz} \cdot 3$$

$$= 2x e^{yz} - x^2 z e^{yz} + 3x^2 y e^{yz}$$

$$= e^{yz}(2x - x^2z + 3x^2y)$$

$$= xe^{yz}(2 - xz + 3xy)$$

### 답

$$\boxed{D_{\mathbf{u}}f(x,y,z) = xe^{yz}(2 - xz + 3xy)}$$

---

## 문제 3. p156. 06 (a) - 합성함수의 도함수

> **문제**: 다음과 같은 함수에 대하여 도함수를 구하세요.
>
> $$z = x^2 + 2y^2 - 3xy, \quad x = \cos t, \quad y = \log t$$

### 풀이

**Step 1: 연쇄 법칙(Chain Rule) 적용**

$z$를 $t$에 대해 미분하기 위해 연쇄 법칙을 사용합니다:

$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}$$

**Step 2: 편미분 계산**

$$\frac{\partial z}{\partial x} = 2x - 3y$$

$$\frac{\partial z}{\partial y} = 4y - 3x$$

**Step 3: $x$와 $y$의 도함수 계산**

$$\frac{dx}{dt} = -\sin t$$

$$\frac{dy}{dt} = \frac{1}{t}$$

**Step 4: 식에 대입**

$$\frac{dz}{dt} = (2x - 3y)(-\sin t) + (4y - 3x)\left(\frac{1}{t}\right)$$

**Step 5: $x = \cos t$, $y = \log t$를 대입**

$$\frac{dz}{dt} = (2\cos t - 3\log t)(-\sin t) + (4\log t - 3\cos t)\left(\frac{1}{t}\right)$$

$$= -2\cos t \sin t + 3\log t \sin t + \frac{4\log t}{t} - \frac{3\cos t}{t}$$

$$= -\sin(2t) + 3\log t \sin t + \frac{4\log t - 3\cos t}{t}$$

### 답

$$\boxed{\frac{dz}{dt} = -\sin(2t) + 3\log t \sin t + \frac{4\log t - 3\cos t}{t}}$$

또는

$$\boxed{\frac{dz}{dt} = -2\cos t \sin t + 3\log t \sin t + \frac{4\log t - 3\cos t}{t}}$$

---

## 문제 4. p157. 09 (a) - 제약조건 하 최적화

> **문제**: 주어진 제약조건에서 다음 함수의 최대값과 최소값을 구하세요.
>
> $$f(x,y) = e^{2xy}, \quad x^2 + y^2 = 4$$

### 풀이

**Step 1: 라그랑주 승수법(Lagrange Multiplier Method) 설정**

제약조건: $g(x,y) = x^2 + y^2 - 4 = 0$

라그랑주 함수:
$$\mathcal{L}(x,y,\lambda) = e^{2xy} - \lambda(x^2 + y^2 - 4)$$

**Step 2: 편미분 = 0 조건**

$$\frac{\partial \mathcal{L}}{\partial x} = 2ye^{2xy} - 2\lambda x = 0 \quad \cdots (1)$$

$$\frac{\partial \mathcal{L}}{\partial y} = 2xe^{2xy} - 2\lambda y = 0 \quad \cdots (2)$$

$$\frac{\partial \mathcal{L}}{\partial \lambda} = -(x^2 + y^2 - 4) = 0 \quad \cdots (3)$$

**Step 3: 식 (1), (2)로부터**

(1): $ye^{2xy} = \lambda x$
(2): $xe^{2xy} = \lambda y$

(1)을 $y$로, (2)를 $x$로 나누면 (단, $x, y \neq 0$인 경우):

$$\frac{e^{2xy}}{1} = \frac{\lambda x}{y}$$
$$\frac{e^{2xy}}{1} = \frac{\lambda y}{x}$$

따라서: $\frac{\lambda x}{y} = \frac{\lambda y}{x}$

$\lambda \neq 0$이면: $x^2 = y^2$, 즉 $y = \pm x$

**Step 4: 경우의 수 분석**

**경우 1**: $y = x$

제약조건에 대입: $x^2 + x^2 = 4 \Rightarrow x^2 = 2 \Rightarrow x = \pm\sqrt{2}$

임계점: $(\sqrt{2}, \sqrt{2})$, $(-\sqrt{2}, -\sqrt{2})$

$$f(\sqrt{2}, \sqrt{2}) = e^{2 \cdot \sqrt{2} \cdot \sqrt{2}} = e^{4}$$
$$f(-\sqrt{2}, -\sqrt{2}) = e^{2 \cdot (-\sqrt{2}) \cdot (-\sqrt{2})} = e^{4}$$

**경우 2**: $y = -x$

제약조건에 대입: $x^2 + x^2 = 4 \Rightarrow x = \pm\sqrt{2}$

임계점: $(\sqrt{2}, -\sqrt{2})$, $(-\sqrt{2}, \sqrt{2})$

$$f(\sqrt{2}, -\sqrt{2}) = e^{2 \cdot \sqrt{2} \cdot (-\sqrt{2})} = e^{-4}$$
$$f(-\sqrt{2}, \sqrt{2}) = e^{2 \cdot (-\sqrt{2}) \cdot \sqrt{2}} = e^{-4}$$

**Step 5: 특수 경우 ($x=0$ 또는 $y=0$)**

- $x = 0$: $y^2 = 4 \Rightarrow y = \pm 2$, $f(0, \pm 2) = e^0 = 1$
- $y = 0$: $x^2 = 4 \Rightarrow x = \pm 2$, $f(\pm 2, 0) = e^0 = 1$

**Step 6: 최대값과 최소값 결정**

$$e^{-4} < 1 < e^4$$

### 답

$$\boxed{\text{최대값: } e^4 \text{ (at } (\sqrt{2}, \sqrt{2}), (-\sqrt{2}, -\sqrt{2}))}$$
$$\boxed{\text{최소값: } e^{-4} \text{ (at } (\sqrt{2}, -\sqrt{2}), (-\sqrt{2}, \sqrt{2}))}$$

---

## 문제 5. p157. 09 (b) - 제약조건 하 최적화

> **문제**: 주어진 제약조건에서 다음 함수의 최대값과 최소값을 구하세요.
>
> $$g(x,y,z) = 2x - y + 3z, \quad x + y - z = 1, \quad x^2 + y^2 = 1$$

### 풀이

**Step 1: 제약조건 정리**

두 개의 제약조건이 있습니다:
- $h_1(x,y,z) = x + y - z - 1 = 0$
- $h_2(x,y,z) = x^2 + y^2 - 1 = 0$

**Step 2: 첫 번째 제약조건으로 $z$ 소거**

$h_1$로부터: $z = x + y - 1$

이를 목적함수에 대입:
$$g(x,y,z) = 2x - y + 3z = 2x - y + 3(x + y - 1)$$
$$= 2x - y + 3x + 3y - 3 = 5x + 2y - 3$$

**Step 3: 단일 제약조건 최적화 문제로 변환**

새로운 목적함수: $f(x,y) = 5x + 2y - 3$
제약조건: $x^2 + y^2 = 1$

**Step 4: 라그랑주 승수법 적용**

$$\mathcal{L}(x,y,\lambda) = 5x + 2y - 3 - \lambda(x^2 + y^2 - 1)$$

편미분:
$$\frac{\partial \mathcal{L}}{\partial x} = 5 - 2\lambda x = 0 \quad \Rightarrow \quad x = \frac{5}{2\lambda}$$

$$\frac{\partial \mathcal{L}}{\partial y} = 2 - 2\lambda y = 0 \quad \Rightarrow \quad y = \frac{1}{\lambda}$$

$$\frac{\partial \mathcal{L}}{\partial \lambda} = -(x^2 + y^2 - 1) = 0$$

**Step 5: $\lambda$ 구하기**

제약조건에 대입:
$$\left(\frac{5}{2\lambda}\right)^2 + \left(\frac{1}{\lambda}\right)^2 = 1$$

$$\frac{25}{4\lambda^2} + \frac{1}{\lambda^2} = 1$$

$$\frac{25 + 4}{4\lambda^2} = 1$$

$$\frac{29}{4\lambda^2} = 1 \quad \Rightarrow \quad \lambda^2 = \frac{29}{4}$$

$$\lambda = \pm\frac{\sqrt{29}}{2}$$

**Step 6: 임계점 계산**

**경우 1**: $\lambda = \frac{\sqrt{29}}{2}$

$$x = \frac{5}{2 \cdot \frac{\sqrt{29}}{2}} = \frac{5}{\sqrt{29}}$$

$$y = \frac{1}{\frac{\sqrt{29}}{2}} = \frac{2}{\sqrt{29}}$$

$$z = x + y - 1 = \frac{5}{\sqrt{29}} + \frac{2}{\sqrt{29}} - 1 = \frac{7}{\sqrt{29}} - 1 = \frac{7 - \sqrt{29}}{\sqrt{29}}$$

목적함수 값:
$$g = 5 \cdot \frac{5}{\sqrt{29}} + 2 \cdot \frac{2}{\sqrt{29}} - 3 = \frac{25 + 4}{\sqrt{29}} - 3 = \frac{29}{\sqrt{29}} - 3 = \sqrt{29} - 3$$

**경우 2**: $\lambda = -\frac{\sqrt{29}}{2}$

$$x = \frac{5}{2 \cdot (-\frac{\sqrt{29}}{2})} = -\frac{5}{\sqrt{29}}$$

$$y = \frac{1}{-\frac{\sqrt{29}}{2}} = -\frac{2}{\sqrt{29}}$$

$$z = -\frac{5}{\sqrt{29}} - \frac{2}{\sqrt{29}} - 1 = -\frac{7}{\sqrt{29}} - 1 = \frac{-7 - \sqrt{29}}{\sqrt{29}}$$

목적함수 값:
$$g = 5 \cdot \left(-\frac{5}{\sqrt{29}}\right) + 2 \cdot \left(-\frac{2}{\sqrt{29}}\right) - 3 = -\frac{29}{\sqrt{29}} - 3 = -\sqrt{29} - 3$$

**Step 7: 최대값과 최소값**

$$\sqrt{29} - 3 > -\sqrt{29} - 3$$

### 답

$$\boxed{\text{최대값: } \sqrt{29} - 3 \text{ (at } x = \frac{5}{\sqrt{29}}, y = \frac{2}{\sqrt{29}}, z = \frac{7-\sqrt{29}}{\sqrt{29}})}$$
$$\boxed{\text{최소값: } -\sqrt{29} - 3 \text{ (at } x = -\frac{5}{\sqrt{29}}, y = -\frac{2}{\sqrt{29}}, z = \frac{-7-\sqrt{29}}{\sqrt{29}})}$$

---

## 문제 6. p158. 13 (a) - Hessian Matrix

> **문제**: 주어진 점에서 다음 함수의 Hessian matrix를 구하세요.
>
> $$f(x,y) = x^2 + 3xy - y^3, \quad (1,2)$$

### 풀이

**Step 1: 1차 편미분 계산**

$$\frac{\partial f}{\partial x} = 2x + 3y$$

$$\frac{\partial f}{\partial y} = 3x - 3y^2$$

**Step 2: 2차 편미분 계산**

$$\frac{\partial^2 f}{\partial x^2} = 2$$

$$\frac{\partial^2 f}{\partial x \partial y} = 3$$

$$\frac{\partial^2 f}{\partial y \partial x} = 3$$

$$\frac{\partial^2 f}{\partial y^2} = -6y$$

**Step 3: Hessian Matrix 구성**

$$H(x,y) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{pmatrix} = \begin{pmatrix}
2 & 3 \\
3 & -6y
\end{pmatrix}$$

**Step 4: 점 (1,2)에서 계산**

$$H(1,2) = \begin{pmatrix}
2 & 3 \\
3 & -6(2)
\end{pmatrix} = \begin{pmatrix}
2 & 3 \\
3 & -12
\end{pmatrix}$$

### 답

$$\boxed{H(1,2) = \begin{pmatrix}
2 & 3 \\
3 & -12
\end{pmatrix}}$$

---

## 문제 7. p158. 13 (b) - Hessian Matrix

> **문제**: 주어진 점에서 다음 함수의 Hessian matrix를 구하세요.
>
> $$g(x,y,z) = e^x \sin y + \log(z^2 + 1), \quad (1,-2,3)$$

### 풀이

**Step 1: 1차 편미분 계산**

$$\frac{\partial g}{\partial x} = e^x \sin y$$

$$\frac{\partial g}{\partial y} = e^x \cos y$$

$$\frac{\partial g}{\partial z} = \frac{2z}{z^2 + 1}$$

**Step 2: 2차 편미분 계산**

$$\frac{\partial^2 g}{\partial x^2} = e^x \sin y$$

$$\frac{\partial^2 g}{\partial x \partial y} = e^x \cos y$$

$$\frac{\partial^2 g}{\partial x \partial z} = 0$$

$$\frac{\partial^2 g}{\partial y \partial x} = e^x \cos y$$

$$\frac{\partial^2 g}{\partial y^2} = -e^x \sin y$$

$$\frac{\partial^2 g}{\partial y \partial z} = 0$$

$$\frac{\partial^2 g}{\partial z \partial x} = 0$$

$$\frac{\partial^2 g}{\partial z \partial y} = 0$$

$$\frac{\partial^2 g}{\partial z^2} = \frac{2(z^2+1) - 2z \cdot 2z}{(z^2+1)^2} = \frac{2z^2 + 2 - 4z^2}{(z^2+1)^2} = \frac{2 - 2z^2}{(z^2+1)^2}$$

**Step 3: Hessian Matrix 구성**

$$H(x,y,z) = \begin{pmatrix}
e^x \sin y & e^x \cos y & 0 \\
e^x \cos y & -e^x \sin y & 0 \\
0 & 0 & \frac{2 - 2z^2}{(z^2+1)^2}
\end{pmatrix}$$

**Step 4: 점 (1,-2,3)에서 계산**

$$e^1 \sin(-2) = e \sin(-2) = -e\sin 2$$

$$e^1 \cos(-2) = e \cos(-2) = e\cos 2$$

$$\frac{2 - 2(3)^2}{(3^2+1)^2} = \frac{2 - 18}{(10)^2} = \frac{-16}{100} = -\frac{4}{25}$$

$$H(1,-2,3) = \begin{pmatrix}
-e\sin 2 & e\cos 2 & 0 \\
e\cos 2 & e\sin 2 & 0 \\
0 & 0 & -\frac{4}{25}
\end{pmatrix}$$

### 답

$$\boxed{H(1,-2,3) = \begin{pmatrix}
-e\sin 2 & e\cos 2 & 0 \\
e\cos 2 & e\sin 2 & 0 \\
0 & 0 & -\frac{4}{25}
\end{pmatrix}}$$

---

## 문제 8. p158. 14 (a) - 극값과 안장점

> **문제**: 다음과 같은 함수의 극대점, 극소점, 안장점을 구하세요.
>
> $$f(x,y) = x^2 + 4y^2 + 2x - 4y - 10$$

### 풀이

**Step 1: 1차 편미분 = 0 (임계점 찾기)**

$$\frac{\partial f}{\partial x} = 2x + 2 = 0 \quad \Rightarrow \quad x = -1$$

$$\frac{\partial f}{\partial y} = 8y - 4 = 0 \quad \Rightarrow \quad y = \frac{1}{2}$$

임계점: $\left(-1, \frac{1}{2}\right)$

**Step 2: 2차 편미분 계산 (Hessian Matrix)**

$$\frac{\partial^2 f}{\partial x^2} = 2$$

$$\frac{\partial^2 f}{\partial x \partial y} = 0$$

$$\frac{\partial^2 f}{\partial y^2} = 8$$

$$H = \begin{pmatrix}
2 & 0 \\
0 & 8
\end{pmatrix}$$

**Step 3: 2차 도함수 판정법**

판별식: $D = f_{xx} f_{yy} - (f_{xy})^2 = 2 \cdot 8 - 0^2 = 16 > 0$

$f_{xx} = 2 > 0$이므로 임계점은 **극소점**입니다.

**Step 4: 극소값 계산**

$$f\left(-1, \frac{1}{2}\right) = (-1)^2 + 4\left(\frac{1}{2}\right)^2 + 2(-1) - 4\left(\frac{1}{2}\right) - 10$$

$$= 1 + 1 - 2 - 2 - 10 = -12$$

### 답

$$\boxed{\text{극소점: } \left(-1, \frac{1}{2}\right), \quad \text{극솟값: } -12}$$
$$\boxed{\text{극대점: 없음, \quad 안장점: 없음}}$$

---

## 문제 9. p158. 14 (b) - 극값과 안장점

> **문제**: 다음과 같은 함수의 극대점, 극소점, 안장점을 구하세요.
>
> $$g(x,y) = 6x^2 - y^3 - 3x^2y + 6y^2 - 3$$

### 풀이

**Step 1: 1차 편미분 = 0 (임계점 찾기)**

$$\frac{\partial g}{\partial x} = 12x - 6xy = 6x(2 - y) = 0$$

$$\frac{\partial g}{\partial y} = -3y^2 - 3x^2 + 12y = -3(y^2 - 4y + x^2) = 0$$

첫 번째 식으로부터: $x = 0$ 또는 $y = 2$

**경우 1**: $x = 0$

두 번째 식에 대입: $-3(y^2 - 4y) = 0 \Rightarrow y(y - 4) = 0$

따라서 $y = 0$ 또는 $y = 4$

임계점: $(0, 0)$, $(0, 4)$

**경우 2**: $y = 2$

두 번째 식에 대입: $-3(4 - 8 + x^2) = 0 \Rightarrow x^2 = 4 \Rightarrow x = \pm 2$

임계점: $(2, 2)$, $(-2, 2)$

**Step 2: 2차 편미분 계산**

$$\frac{\partial^2 g}{\partial x^2} = 12 - 6y$$

$$\frac{\partial^2 g}{\partial x \partial y} = -6x$$

$$\frac{\partial^2 g}{\partial y^2} = -6y + 12$$

**Step 3: 각 임계점에서 판정**

**임계점 (0, 0):**

$$H(0,0) = \begin{pmatrix}
12 & 0 \\
0 & 12
\end{pmatrix}$$

$D = 12 \cdot 12 - 0 = 144 > 0$, $g_{xx} = 12 > 0$ → **극소점**

$$g(0,0) = -3$$

**임계점 (0, 4):**

$$H(0,4) = \begin{pmatrix}
12 - 24 & 0 \\
0 & -24 + 12
\end{pmatrix} = \begin{pmatrix}
-12 & 0 \\
0 & -12
\end{pmatrix}$$

$D = (-12)(-12) - 0 = 144 > 0$, $g_{xx} = -12 < 0$ → **극대점**

$$g(0,4) = 0 - 64 - 0 + 96 - 3 = 29$$

**임계점 (2, 2):**

$$H(2,2) = \begin{pmatrix}
12 - 12 & -12 \\
-12 & -12 + 12
\end{pmatrix} = \begin{pmatrix}
0 & -12 \\
-12 & 0
\end{pmatrix}$$

$D = 0 \cdot 0 - (-12)^2 = -144 < 0$ → **안장점**

**임계점 (-2, 2):**

$$H(-2,2) = \begin{pmatrix}
0 & 12 \\
12 & 0
\end{pmatrix}$$

$D = 0 \cdot 0 - 12^2 = -144 < 0$ → **안장점**

### 답

$$\boxed{\text{극소점: } (0, 0), \quad \text{극솟값: } -3}$$
$$\boxed{\text{극대점: } (0, 4), \quad \text{극댓값: } 29}$$
$$\boxed{\text{안장점: } (2, 2), (-2, 2)}$$

---

## 문제 10. p158. 15 (b) - 야코비 Matrix

> **문제**: 다음 변환의 야코비 matrix를 구하세요.
>
> $$x = u\sin v, \quad y = u\cos v$$

### 풀이

**Step 1: 야코비 행렬의 정의**

변환 $(u, v) \rightarrow (x, y)$의 야코비 행렬은 다음과 같이 정의됩니다:

$$J = \begin{pmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v}
\end{pmatrix}$$

**Step 2: 각 편미분 계산**

$$\frac{\partial x}{\partial u} = \sin v$$

$$\frac{\partial x}{\partial v} = u\cos v$$

$$\frac{\partial y}{\partial u} = \cos v$$

$$\frac{\partial y}{\partial v} = -u\sin v$$

**Step 3: 야코비 행렬 구성**

$$J = \begin{pmatrix}
\sin v & u\cos v \\
\cos v & -u\sin v
\end{pmatrix}$$

**Step 4: 야코비안(Jacobian determinant) 계산 (참고)**

$$\det(J) = \sin v \cdot (-u\sin v) - u\cos v \cdot \cos v$$

$$= -u\sin^2 v - u\cos^2 v = -u(\sin^2 v + \cos^2 v) = -u$$

### 답

$$\boxed{J = \begin{pmatrix}
\sin v & u\cos v \\
\cos v & -u\sin v
\end{pmatrix}}$$

야코비안: $\boxed{\det(J) = -u}$

---
