// ----------------------------
// 이 예제는 OpenGL에서 삼각함수를 이용하여 원 모양으로 점을 찍는 예제입니다.
// (참고: 모든 점이 보이지 않으면 glOrtho의 zNear, zFar 값을 충분히 넓게 잡아야 합니다.)
// ----------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>    // 삼각함수 사용을 위한 헤더

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0f, 0.0f, 0.0f); // 빨간색
    glPointSize(2.0f);

    glBegin(GL_POINTS);
    // 원 위의 점들을 삼각함수로 계산
    for (float alpha = 0; alpha < 2 * 3.14f; alpha += 0.5f) {
        // float alpha = 2.0f * 3.14f * float(i) / 20.0f;
        float x = 10.0f * cos(alpha);
        float y = 10.0f * sin(alpha);
        float z = 0.0f; 
        glVertex3f(x, y, z);
    }
    glEnd();

    glFlush();
}

void SetupRc(void)
{
    std::cout << "SetupRc" << std::endl;
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
}

void ChangeSize(GLsizei w, GLsizei h) {
    GLint wSize = 100;    // 가상의 논리 좌표계 절반 길이
    GLfloat aspectRatio;

    if (h == 0) {
        h = 1;
    }
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    aspectRatio = (GLfloat)w / (GLfloat)h;

    // zNear/zFar 값을 넉넉하게 (-100 ~ +100)로 해서 모든 점이 보이게 함
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
    glutCreateWindow("원 위에 점 찍기 (Draw Points on a Circle)");

    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);

    glutMainLoop();
    return 0;
}