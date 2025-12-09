// GL_TRIANGLE_FAN을 이용한 솔리드 원뿔 그리기
// - 원뿔 옆면: GL_TRIANGLE_FAN (꼭지점 + 원주 점들)
// - 원뿔 밑면: GL_TRIANGLE_FAN (중심점 + 원주 점들)
// - 키보드로 회전 가능 (방향키)

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

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);   // 후면 제거 활성화

    glPushMatrix();
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    GLfloat angle, x, y;
    GLfloat radius = 50.0f;   // 밑면 반지름
    GLfloat height = 80.0f;   // 원뿔 높이
    int segments = 16;         // 분할 개수

    // ===== 1. 원뿔 옆면 (GL_TRIANGLE_FAN) =====
    // 꼭지점을 중심으로 밑면 원주 점들을 연결하여 삼각형 면 생성
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
        glVertex3f(0.0f, 0.0f, height);  // 꼭지점 (중심)

        // 밑면 원주를 따라 점들을 생성 (반시계 방향)
        for (int i = 0; i <= segments; i++) {
            angle = 2.0f * PI * i / segments;
            x = radius * cos(angle);
            y = radius * sin(angle);

            // 각 삼각형마다 색상 교차
            if (i % 2 == 0) {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            } else {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            }

            glVertex3f(x, y, 0.0f);
        }
    glEnd();

    // ===== 2. 원뿔 밑면 (GL_TRIANGLE_FAN) =====
    // 중심점을 기준으로 원주 점들을 연결하여 원 형태 생성
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
        glVertex3f(0.0f, 0.0f, 0.0f);  // 밑면 중심점

        // 밑면 원주를 따라 점들을 생성 (시계 방향, 밑에서 보이도록)
        for (int i = 0; i <= segments; i++) {
            angle = 2.0f * PI * i / segments;
            x = radius * cos(angle);
            y = radius * sin(angle);

            if (i % 2 == 0) {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            } else {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            }

            glVertex3f(x, -y, 0.0f);  // y축 반전 (시계 방향)
        }
    glEnd();

    glPopMatrix();
    glutSwapBuffers();
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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(400, 100);
    glutCreateWindow("Solid Cone - GL_TRIANGLE_FAN");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutSpecialFunc(SpecialKeys);

    std::cout << "=========================================" << std::endl;
    std::cout << "GL_TRIANGLE_FAN으로 솔리드 원뿔 그리기" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "[회전] 방향키 (↑↓←→)" << std::endl;
    std::cout << "=========================================" << std::endl;

    glutMainLoop();

    return 0;
}
