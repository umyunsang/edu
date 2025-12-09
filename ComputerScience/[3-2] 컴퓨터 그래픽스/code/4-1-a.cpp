// 4-1-a: 3D 지오메트리 실습 - 회전하는 정육면체
// - 원근 투영, 카메라, 깊이 테스트, 후면 제거, 더블 버퍼링, 키보드 상호작용 포함

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

// 회전 각도를 위한 전역 변수
GLfloat xRot = 0.0f;
GLfloat yRot = 0.0f;

// 정육면체의 8개 정점 좌표
GLfloat vertices[][3] = {
    {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
    {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}
};

// 정육면체의 6개 면을 정의하는 정점 인덱스 (반시계 방향)
GLint faces[][4] = {
    {0, 3, 2, 1}, // 뒷면
    {0, 4, 7, 3}, // 왼쪽
    {1, 2, 6, 5}, // 오른쪽
    {4, 5, 6, 7}, // 앞면
    {3, 7, 6, 2}, // 윗면
    {0, 1, 5, 4}  // 아랫면
};

// 각 면의 색상
GLfloat colors[][3] = {
    {0.0, 0.0, 1.0}, // Blue
    {1.0, 0.0, 0.0}, // Red
    {0.0, 1.0, 0.0}, // Green
    {1.0, 1.0, 0.0}, // Yellow
    {1.0, 0.0, 1.0}, // Magenta
    {0.0, 1.0, 1.0}  // Cyan
};

// 정육면체를 그리는 함수
void DrawCube(void) {
    for (int i = 0; i < 6; i++) {
        glBegin(GL_QUADS);
            glColor3fv(colors[i]);
            for (int j = 0; j < 4; j++) {
                glVertex3fv(vertices[faces[i][j]]);
            }
        glEnd();
    }
}

// 화면 렌더링 콜백 함수
void RenderScene(void) {
    // 컬러 버퍼와 깊이 버퍼를 초기화
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ModelView 행렬 스택에 현재 상태 저장
    glPushMatrix();

    // 카메라 위치와 방향 설정
    gluLookAt(0.0, 0.0, 5.0,   // 카메라 위치 (0, 0, 5)
              0.0, 0.0, 0.0,   // 바라보는 지점 (원점)
              0.0, 1.0, 0.0);  // 상향 벡터 (Y축)

    // 회전 변환 적용
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    // 정육면체 그리기
    DrawCube();

    // ModelView 행렬 스택에서 이전 상태 복원
    glPopMatrix();

    // 백 버퍼와 프론트 버퍼를 교체하여 화면에 표시
    glutSwapBuffers();
}

// 초기화 함수
void SetupRC(void) {
    // 배경색 설정 (검은색)
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // 깊이 테스트 활성화
    glEnable(GL_DEPTH_TEST);
    // 후면 제거 활성화
    glEnable(GL_CULL_FACE);
    // 셰이딩 모델 설정 (스무스)
    glShadeModel(GL_SMOOTH);
}

// 특수 키 입력 처리 콜백 함수
void SpecialKeys(int key, int x, int y) {
    if (key == GLUT_KEY_UP) xRot -= 5.0f;
    if (key == GLUT_KEY_DOWN) xRot += 5.0f;
    if (key == GLUT_KEY_LEFT) yRot -= 5.0f;
    if (key == GLUT_KEY_RIGHT) yRot += 5.0f;

    // 회전 각도 범위 제한
    if (xRot > 360.0f) xRot = 0.0f;
    if (xRot < -360.0f) xRot = 0.0f;
    if (yRot > 360.0f) yRot = 0.0f;
    if (yRot < -360.0f) yRot = 0.0f;

    // 화면 갱신 요청
    glutPostRedisplay();
}

// 창 크기 변경 콜백 함수
void ChangeSize(GLsizei w, GLsizei h) {
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);

    // Projection 행렬 설정
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 원근 투영 설정
    GLfloat aspectRatio = (GLfloat)w / (GLfloat)h;
    gluPerspective(45.0f, aspectRatio, 1.0f, 100.0f);

    // ModelView 행렬 모드로 복귀
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    // 디스플레이 모드 설정 (더블 버퍼, RGB 컬러, 깊이 버퍼)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Rotating Cube");

    // 콜백 함수 등록
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutSpecialFunc(SpecialKeys);

    // 초기화 함수 호출
    SetupRC();

    // GLUT 이벤트 루프 시작
    glutMainLoop();

    return 0;
}