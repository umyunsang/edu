#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 0.0f, 0.0f);

    GLfloat Width = 10.0f;
    GLfloat x = 80.0f;

    // GL_LINES: 두 개의 정점을 연결하여 독립적인 선을 그림
    for(int i = 0; i < 10; i++) {
        glLineWidth(Width);
        glBegin(GL_LINES);
        // x값이 고정, y값만 변경 → 세로 선 그리기
        glVertex3f(x, -80.0f, 0.0f);  
        glVertex3f(x, 80.0f, 0.0f);   
        glEnd();

        Width -= 1.0f;  // 선 두께 1씩 감소
        x -= 18.0f;     // x좌표를 왼쪽으로 이동
    }

    glFlush();
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

    glutCreateWindow("GL_LINES");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glutDisplayFunc(RenderScene);

    glutReshapeFunc(ChangeSize);

    glutMainLoop();

    return 0;
}
