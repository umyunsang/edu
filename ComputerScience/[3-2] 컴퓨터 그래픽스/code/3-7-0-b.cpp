// GL_TRIANGLE_FAN을 이용한 원 조각 그리기
// - 키보드 1~8을 누르면 원 조각이 하나씩 생성됨
// - 색상: 초록, 빨강 순서대로 교차
// - 총 8조각으로 완전한 원 구성

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

int pieceCount = 0;  // 현재 그려진 조각 개수 (0~8)

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    GLfloat angle, x, y;
    GLfloat radius = 50.0f;

    // pieceCount만큼만 조각 그리기
    for (int i = 0; i < pieceCount; i++) {
        glBegin(GL_TRIANGLES);
            // 색상 설정: 초록, 빨강 순서대로 교차
            if (i % 2 == 0) {
                glColor3f(0.0f, 1.0f, 0.0f);  // 초록색
            } else {
                glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
            }

            // 중심점
            glVertex2f(0.0f, 0.0f);

            // 첫 번째 원주 점
            angle = 2.0f * PI * i / 8.0f;
            x = radius * cos(angle);
            y = radius * sin(angle);
            glVertex2f(x, y);

            // 두 번째 원주 점
            angle = 2.0f * PI * (i + 1) / 8.0f;
            x = radius * cos(angle);
            y = radius * sin(angle);
            glVertex2f(x, y);
        glEnd();
    }

    glFlush();
}

void Keyboard(unsigned char key, int x, int y) {
    // 키보드 1~8 입력 처리
    if (key >= '1' && key <= '8') {
        int targetPiece = key - '0';  // '1' -> 1, '2' -> 2, ...
        pieceCount = targetPiece;
        std::cout << "조각 개수: " << pieceCount << "/8" << std::endl;
        glutPostRedisplay();
    }
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
    glutCreateWindow("Circle Pieces - Press 1~8");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutKeyboardFunc(Keyboard);

    std::cout << "=========================================" << std::endl;
    std::cout << "원 조각 그리기 (키보드 1~8)" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "1번 키: 1조각 (초록)" << std::endl;
    std::cout << "2번 키: 2조각 (초록, 빨강)" << std::endl;
    std::cout << "3번 키: 3조각 (초록, 빨강, 초록)" << std::endl;
    std::cout << "..." << std::endl;
    std::cout << "8번 키: 8조각 (완전한 원)" << std::endl;
    std::cout << "=========================================" << std::endl;

    glutMainLoop();

    return 0;
}
