#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

GLfloat len = 50.0f;
GLint factor = 1;
GLushort pattern = 0x5555;

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glPushMatrix();
    glRotatef(45, 1.0f, 0.0f, 0.0f);
    glRotatef(45, 0.0f, 1.0f, 0.0f);

    glLineWidth(2.0f);

    // 보이지 않는 뒷면 모서리 (점선)
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(factor, pattern);
    glBegin(GL_LINES);
    // 뒤쪽 왼쪽 세로선
    glVertex3f(-len, -len, -len);
    glVertex3f(-len, len, -len);
    // 뒤쪽 위 가로선
    glVertex3f(-len, len, -len);
    glVertex3f(len, len, -len);
    // 왼쪽 아래 깊이선
    glVertex3f(-len, -len, len);
    glVertex3f(-len, -len, -len);
    glEnd();
    glDisable(GL_LINE_STIPPLE);

    // 보이는 모서리 (실선)
    glBegin(GL_LINES);
    // 앞면 사각형
    glVertex3f(-len, -len, len);
    glVertex3f(len, -len, len);
    glVertex3f(len, -len, len);
    glVertex3f(len, len, len);
    glVertex3f(len, len, len);
    glVertex3f(-len, len, len);
    glVertex3f(-len, len, len);
    glVertex3f(-len, -len, len);

    // 오른쪽 면 세로선 2개
    glVertex3f(len, -len, len);
    glVertex3f(len, -len, -len);
    glVertex3f(len, len, len);
    glVertex3f(len, len, -len);

    // 아래 면 가로선
    glVertex3f(-len, -len, -len);
    glVertex3f(len, -len, -len);

    // 위 면 깊이선
    glVertex3f(-len, len, len);
    glVertex3f(-len, len, -len);
    glVertex3f(len, len, len);
    glVertex3f(len, len, -len);

    // 뒤쪽 오른쪽 세로선
    glVertex3f(len, -len, -len);
    glVertex3f(len, len, -len);
    glEnd();
    glPopMatrix();
    glFlush();
}

void SetupRc(void)
{
    std::cout << "OpenGL Basic Triangle Example" << std::endl;
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 흰색 배경
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

    glutCreateWindow("Basic Triangle Example");

    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

    glutDisplayFunc(RenderScene);

    glutReshapeFunc(ChangeSize);

    glutMainLoop();

    return 0;
}