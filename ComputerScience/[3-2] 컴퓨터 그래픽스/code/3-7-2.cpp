// GL_TRIANGLE_FAN을 이용한 솔리드 원뿔 그리기 (깊이 버퍼 사용)
// - 깊이 버퍼: 가려진 면을 자동으로 제거
// - 픽셀이 그려지면 깊이 값(관측자로부터의 거리)을 할당
// - 같은 위치에 다른 픽셀이 그려질 때 깊이 값을 비교
// - 낮은 깊이 값(더 가까운)을 가진 픽셀만 화면에 표시
// - GL_DEPTH_TEST를 활성화하고 GLUT_DEPTH 모드 사용 필요
// - 더블 버퍼 사용으로 부드러운 렌더링

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

GLfloat xRot = 0.0f;
GLfloat yRot = 0.0f;
GLboolean bDepth = GL_TRUE;  // 깊이 테스트 on/off 플래그

void RenderScene(void) {
    // 컬러 버퍼와 깊이 버퍼 모두 클리어
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 깊이 테스트 on/off 토글
    if (bDepth)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);   // 후면 제거 활성화

    glPushMatrix();
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    GLfloat angle, x, y;
    GLfloat radius = 50.0f;   // 밑면 반지름
    GLfloat height = 80.0f;   // 원뿔 높이
    int count = 0;

    // ===== 1. 원뿔 옆면 (GL_TRIANGLE_FAN) =====
    // 꼭지점을 중심으로 밑면 원주 점들을 연결하여 삼각형 면 생성
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, 0.0f, height);  // 꼭지점 (중심)

        for (angle = 0.0f; angle <= 2.0f * PI + 0.01f; angle += (2.0f * PI) / 16.0f) {
            if (count % 2 == 0) {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            } else {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            }

            x = radius * cos(angle);
            y = radius * sin(angle);
            glVertex3f(x, y, 0.0f);
            count++;
        }
    glEnd();

    // ===== 2. 원뿔 밑면 (GL_TRIANGLE_FAN) =====
    // 중심점을 기준으로 원주 점들을 연결하여 원 형태 생성
    count = 0;
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, 0.0f, 0.0f);  // 밑면 중심점

        for (angle = 0.0f; angle <= 2.0f * PI + 0.01f; angle += (2.0f * PI) / 16.0f) {
            if (count % 2 == 0) {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            } else {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            }

            x = radius * cos(angle);
            y = radius * sin(angle);
            glVertex3f(x, -y, 0.0f);  // y축 반전 (시계 방향)
            count++;
        }
    glEnd();

    glPopMatrix();

    // 더블 버퍼링: 백 버퍼와 프론트 버퍼를 교체
    // 화면 깜빡임 없이 부드러운 렌더링 가능
    glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y) {
    if (key == 'd' || key == 'D') {
        bDepth = !bDepth;  // 깊이 테스트 토글
        if (bDepth)
            std::cout << "깊이 테스트: 활성화" << std::endl;
        else
            std::cout << "깊이 테스트: 비활성화" << std::endl;
    }

    glutPostRedisplay();
}

void SpecialKeys(int key, int x, int y) {
    if(key == GLUT_KEY_UP)
        xRot -= 5.0f;

    if(key == GLUT_KEY_DOWN)
        xRot += 5.0f;

    if(key == GLUT_KEY_LEFT)
        yRot -= 5.0f;

    if(key == GLUT_KEY_RIGHT)
        yRot += 5.0f;

    if(xRot > 360.0f)
        xRot = 0.0f;
    if(xRot < -360.0f)
        xRot = 0.0f;

    if(yRot > 360.0f)
        yRot = 0.0f;
    if(yRot < -360.0f)
        yRot = 0.0f;

    glutPostRedisplay();
}

void ChangeSize(GLsizei w, GLsizei h) {
    GLfloat aspectRatio;

    if (h == 0) {
        h = 1;
    }

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    aspectRatio = (GLfloat)w / (GLfloat)h;

    // 3D 원근 투영 설정
    gluPerspective(45.0f, aspectRatio, 1.0f, 500.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 카메라 위치 설정 (약간 위에서 바라봄)
    glTranslatef(0.0f, -30.0f, -250.0f);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);

    // GLUT_DEPTH: 깊이 버퍼 활성화 (z-buffering)
    // GLUT_DOUBLE: 더블 버퍼링 활성화 (화면 깜빡임 방지)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(600, 600);
    glutInitWindowPosition(400, 100);
    glutCreateWindow("Solid Cone with Depth Buffer");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutKeyboardFunc(Keyboard);
    glutSpecialFunc(SpecialKeys);

    std::cout << "=========================================" << std::endl;
    std::cout << "깊이 버퍼를 사용한 솔리드 원뿔 그리기" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "[회전] 방향키 (↑↓←→)" << std::endl;
    std::cout << "[깊이 테스트 토글] d 또는 D 키" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "깊이 버퍼 기능:" << std::endl;
    std::cout << "- GL_DEPTH_TEST: 깊이 값 비교 활성화" << std::endl;
    std::cout << "- GLUT_DEPTH: 깊이 버퍼 할당" << std::endl;
    std::cout << "- GL_DEPTH_BUFFER_BIT: 깊이 버퍼 클리어" << std::endl;
    std::cout << "- 가까운 픽셀만 화면에 표시됨" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "깊이 테스트: 활성화" << std::endl;

    glutMainLoop();

    return 0;
}
