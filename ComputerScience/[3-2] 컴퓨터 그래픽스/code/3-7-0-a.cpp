// GL_TRIANGLE_FAN을 이용한 원 그리기
// - 원점 포함 17점 (16조각) 사용
// - 각 삼각형 면은 빨간색/초록색 교차
// - 키보드로 원 이동 가능 (a/d/w/x)

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

GLboolean bCull = GL_TRUE;   // 후면 제거 활성화 (앞/뒤 확인용)
GLfloat xTran = 0.0f;
GLfloat yTran = 0.0f;
GLfloat xRot = 0.0f;
GLfloat yRot = 0.0f;

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    glPushMatrix();
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
    glTranslatef(xTran, yTran, 0.0f);

    GLfloat angle, x, y;
    int count = 0;

    // glColor는 현재 면에만 영향을 주므로,
    // 앞면과 뒷면에 각각 다른 색상을 지정
    // 앞면 (CCW, 반시계방향)
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(0.0f, 0.0f);  // 중심점

        for (angle = 0.0f; angle <= 2.0f * PI + 0.01f; angle += PI / 8.0f) {
            if (count % 2 == 0) {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            } else {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            }

            x = 50.0f * cos(angle);
            y = 50.0f * sin(angle);
            glVertex2f(x, y);
            count++;
        }
    glEnd();

    // 뒷면 (CW, 시계방향)
    count = 0;
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(0.0f, 0.0f);  // 중심점

        for (angle = 2.0f * PI; angle >= -0.01f; angle -= PI / 8.0f) {
            if (count % 2 == 0) {
                glColor3f(0.0f, 0.0f, 1.0f);  // 파란색
            } else {
                glColor3f(1.0f, 1.0f, 0.0f);  // 노란색
            }

            x = 50.0f * cos(angle);
            y = 50.0f * sin(angle);
            glVertex2f(x, y);
            count++;
        }
    glEnd();

    glPopMatrix();
    glFlush();
}

// ============================================================
// 특수 키(방향키) 이벤트 처리 함수
// ============================================================
void SpecialKeys(int key, int x, int y) {
    // 1. 방향키 입력에 따라 회전 각도 변경
    if(key == GLUT_KEY_UP)
        xRot -= 2.0f;  // ↑ 키: X축 기준 위로 회전

    if(key == GLUT_KEY_DOWN)
        xRot += 2.0f;  // ↓ 키: X축 기준 아래로 회전

    if(key == GLUT_KEY_LEFT)
        yRot -= 2.0f;  // ← 키: Y축 기준 왼쪽 회전

    if(key == GLUT_KEY_RIGHT)
        yRot += 2.0f;  // → 키: Y축 기준 오른쪽 회전

    // 2. 각도 범위 제한 (0 ~ 360도)
    if(xRot > 360.0f)
        xRot = 0.0f;
    if(xRot < -360.0f)
        xRot = 0.0f;

    if(yRot > 360.0f)
        yRot = 0.0f;
    if(yRot < -360.0f)
        yRot = 0.0f;

    // 3. 화면 갱신 요청
    glutPostRedisplay();

    // ★ 중요 포인트 ★
    // 1) 특수 키 콜백은 glutSpecialFunc()로 등록해야 작동함
    // 2) 일반 키보드(Keyboard 함수)와 특수 키(SpecialKeys 함수)는 별도로 등록
    // 3) 특수 키 상수: GLUT_KEY_UP, GLUT_KEY_DOWN, GLUT_KEY_LEFT, GLUT_KEY_RIGHT
    // 4) glRotatef(각도, x, y, z): (x, y, z) 벡터를 축으로 회전
    //    - glRotatef(xRot, 1, 0, 0): X축 기준 회전
    //    - glRotatef(yRot, 0, 1, 0): Y축 기준 회전
}

// ============================================================
// 키보드 이벤트 처리 함수
// ============================================================
void Keyboard(unsigned char key, int x, int y) {
    // 1. 키 입력에 따라 전역 변수 xTran, yTran 값 변경
    if(key == 'a') {
        xTran -= 2.0f;  // a 키: 왼쪽 이동 (-X 방향)
    } else if(key == 'd') {
        xTran += 2.0f;  // d 키: 오른쪽 이동 (+X 방향)
    } else if(key == 'w') {
        yTran += 2.0f;  // w 키: 위로 이동 (+Y 방향)
    } else if(key == 'x') {
        yTran -= 2.0f;  // x 키: 아래로 이동 (-Y 방향)
    }

    // 2. glutPostRedisplay() 호출: 화면 갱신 요청
    // - 이 함수는 GLUT에게 "다음 기회에 RenderScene()을 다시 호출해줘"라고 요청
    // - 즉시 실행되는 것이 아니라, 다음 이벤트 루프 사이클에서 실행됨
    // - 이 호출이 없으면 화면이 갱신되지 않아 이동이 보이지 않음!
    glutPostRedisplay();

    // ★ 중요 포인트 ★
    // 1) 키보드 콜백은 glutKeyboardFunc()로 등록해야 작동함
    //    (main 함수의 109번 줄 참조)
    // 2) 콜백 등록은 반드시 glutInit() 이후, glutMainLoop() 이전에 해야 함
    // 3) glutPostRedisplay()를 호출하지 않으면 변수는 변경되지만 화면은 갱신 안 됨
    // 4) 매개변수 x, y는 마우스 커서의 위치 (이 예제에서는 미사용)
}

void ChangeSize(GLsizei w, GLsizei h) {
    GLint wSize = 100;
    GLfloat aspectRatio;

    if (h == 0) {
        h = 1;
    }

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    aspectRatio = (GLfloat)w / (GLfloat)h;

    if (w <= h) {
        glOrtho(-wSize, wSize, -wSize / aspectRatio, wSize / aspectRatio, -100, 100);
    } else {
        glOrtho(-wSize * aspectRatio, wSize * aspectRatio, -wSize, wSize, -100, 100);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(400, 400);
    glutCreateWindow("GL_TRIANGLE_FAN Circle");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);

    // 콜백 함수 등록
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutKeyboardFunc(Keyboard);      // 일반 키보드 콜백 등록
    glutSpecialFunc(SpecialKeys);    // 특수 키(방향키) 콜백 등록 (필수!)

    std::cout << "GL_TRIANGLE_FAN으로 원 그리기 + 후면 제거" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "[이동] a(왼쪽), d(오른쪽), w(위), x(아래)" << std::endl;
    std::cout << "[회전] 방향키 (↑↓←→)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "앞면: 빨강/초록 교차" << std::endl;
    std::cout << "뒷면: 파랑/노랑 교차" << std::endl;
    std::cout << "==========================================" << std::endl;

    glutMainLoop();

    return 0;
}
