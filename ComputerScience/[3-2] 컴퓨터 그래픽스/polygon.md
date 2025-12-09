# OpenGL 폴리곤 그리기 정리

## 기본 개념

### 폴리곤 정점 순서 (Vertex Order)
OpenGL에서는 정점을 정의하는 순서가 매우 중요합니다.

- **반시계 방향 (CCW, Counter-Clockwise)**: 정면 (Front Face)
- **시계 방향 (CW, Clockwise)**: 뒷면 (Back Face)

## 폴리곤 모드 설정

### glPolygonMode()
폴리곤을 어떻게 렌더링할지 결정합니다.

**함수 원형:**
```cpp
void glPolygonMode(GLenum face, GLenum mode);
```

**매개변수:**
- `face`:
  - `GL_FRONT` - 정면만
  - `GL_BACK` - 뒷면만
  - `GL_FRONT_AND_BACK` - 양면

- `mode`:
  - `GL_FILL` - 채우기 (기본값)
  - `GL_LINE` - 와이어프레임
  - `GL_POINT` - 정점만

## 면 제거 (Face Culling)

성능 향상을 위해 보이지 않는 면을 렌더링하지 않을 수 있습니다.

**관련 함수:**
- `glEnable(GL_CULL_FACE)` - 면 제거 활성화
- `glCullFace(mode)` - 제거할 면 설정
  - `GL_BACK` - 뒷면 제거 (기본값)
  - `GL_FRONT` - 정면 제거
  - `GL_FRONT_AND_BACK` - 양면 제거
- `glFrontFace(mode)` - 정면 방향 정의
  - `GL_CCW` - 반시계 방향이 정면 (기본값)
  - `GL_CW` - 시계 방향이 정면

## 폴리곤 그리기 모드

### GL_TRIANGLES
독립적인 삼각형들을 그립니다 (3개 정점당 1개 삼각형).

### GL_TRIANGLE_STRIP
연결된 삼각형 띠를 그립니다.
- 처음 3개 정점으로 첫 삼각형
- 이후 정점마다 새 삼각형 추가

### GL_TRIANGLE_FAN
부채꼴 모양의 삼각형들을 그립니다.
- 첫 정점이 중심점 (모든 삼각형의 공통 정점)
- 이후 정점들이 부채꼴 형태로 삼각형 생성

### GL_QUADS
독립적인 사각형들을 그립니다 (4개 정점당 1개 사각형).

### GL_POLYGON
단일 볼록 다각형을 그립니다.
- 주의: 볼록 다각형만 지원

## 폴리곤 색상 및 셰이딩

### glShadeModel()
셰이딩 방식을 설정합니다.

**함수 원형:**
```cpp
void glShadeModel(GLenum mode);
```

**매개변수:**
- `GL_FLAT` - 플랫 셰이딩
- `GL_SMOOTH` - 스무스 셰이딩 (구로 셰이딩, Gouraud Shading) **(기본값)**

### GL_FLAT (플랫 셰이딩)

폴리곤 전체가 **하나의 색상**으로 채워집니다.

**특징:**
- 마지막 정점의 색상이 폴리곤 전체에 적용됩니다
- 각진 느낌의 렌더링
- 성능이 더 빠름
- 사용 시나리오: 만화 스타일, 로우폴리 모델, UI 요소

### GL_SMOOTH (스무스 셰이딩)

각 정점의 색상을 **보간(interpolation)**하여 그라데이션 효과를 만듭니다.

**특징:**
- 정점 간 색상이 부드럽게 전환됩니다
- 더 사실적인 렌더링
- 약간의 성능 비용 발생
- 사용 시나리오: 사실적인 3D 모델, 그라데이션 효과

### 색상 지정 함수

**RGB 색상:**
- `glColor3f(r, g, b)` - RGB, 각 값 0.0 ~ 1.0
- `glColor3ub(r, g, b)` - RGB, 각 값 0 ~ 255

**RGBA 색상 (투명도 포함):**
- `glColor4f(r, g, b, a)` - RGBA, 각 값 0.0 ~ 1.0
- `glColor4ub(r, g, b, a)` - RGBA, 각 값 0 ~ 255

### 투명도 (Alpha Blending)

투명도를 사용하려면:
1. `glEnable(GL_BLEND)` - 블렌딩 활성화
2. `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)` - 블렌딩 함수 설정
3. `glColor4f(r, g, b, alpha)` - Alpha 값 포함한 색상 지정
4. `glDisable(GL_BLEND)` - 블렌딩 비활성화

## 주의사항

1. **정점 순서 일관성**: 모든 폴리곤의 정점을 같은 방향(반시계)으로 정의해야 합니다.

2. **볼록 다각형**: `GL_POLYGON`은 볼록 다각형만 지원합니다. 오목 다각형은 삼각형으로 분해해야 합니다.

3. **성능 최적화**:
   - 면 제거(`glEnable(GL_CULL_FACE)`)로 뒷면을 렌더링하지 않아 성능 향상
   - Triangle strip/fan으로 정점 재사용

4. **깊이 테스트**: 3D 객체를 올바르게 그리려면 깊이 버퍼가 필요합니다.
   - `glEnable(GL_DEPTH_TEST)` 사용

5. **와인딩 순서 확인**: 카메라 위치나 회전에 따라 정면/뒷면이 바뀔 수 있으니 정점 순서를 신중하게 정의해야 합니다.

6. **셰이딩 모드**: Flat과 Smooth 셰이딩의 차이를 이해하고 상황에 맞게 선택하세요.
