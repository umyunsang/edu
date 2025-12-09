#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

void RenderScene(void) {
    GLfloat y;
    GLint factor = 1;
    GLushort pattern = 0x5555;  // 0101010101010101 (1=그림, 0=안그림)

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 0.0f, 0.0f);
    glPushMatrix();

    // glEnable(GL_LINE_STIPPLE): 점선 그리기 기능 활성화
    glEnable(GL_LINE_STIPPLE);

    // y좌표 -90부터 90까지 20씩 증가하며 10개의 가로선 그리기
    for(y = -90.0f; y <= 90.0f; y += 20.0f) {
        // glLineStipple(factor, pattern)
        // - factor: 패턴 각 비트를 몇 번 반복할지 (값이 클수록 점선 간격이 넓어짐)
        // - pattern: 16비트 점선 패턴 (0x5555 = 0101010101010101)
        //   각 선마다 factor가 1씩 증가하여 점선 간격이 점점 넓어짐
        glLineStipple(factor, pattern);
        glBegin(GL_LINES);
        glVertex3f(-80.0f, y, 0.0f);  // 선의 시작점 (왼쪽)
        glVertex3f(80.0f, y, 0.0f);   // 선의 끝점 (오른쪽)
        glEnd();
        factor++;  // 다음 선은 factor가 증가하여 간격이 더 넓어짐
    }

    // glDisable(GL_LINE_STIPPLE): 점선 그리기 기능 비활성화
    glDisable(GL_LINE_STIPPLE);

    glPopMatrix();
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

    glutCreateWindow("GL_LINE_STIPPLE");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glutDisplayFunc(RenderScene);

    glutReshapeFunc(ChangeSize);

    glutMainLoop();

    return 0;
}
