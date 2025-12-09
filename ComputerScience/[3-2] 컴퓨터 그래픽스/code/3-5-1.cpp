#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

void RenderScene(void) {
    GLfloat y;
    GLint factor = 3;
    GLushort pattern = 0x00FF;
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 0.0f, 0.0f);
    glPushMatrix();

    glEnable(GL_LINE_STIPPLE);

    // y좌표 -90부터 90까지 20씩 증가하며 10개의 가로선 그리기
    // factor는 고정(3), pattern을 곱하기 연산으로 변경하여 듬성듬성한 점선에서 촘촘해지게 함
    for(y = -90.0f; y <= 90.0f; y += 20.0f) {
        // glLineStipple(factor, pattern)
        // - factor: 3으로 고정 (패턴 확대 배율)
        // - pattern: 반복마다 2배 증가 → 1비트가 추가되어 점선이 촘촘해짐
        //   0x0101 → 0x0202 → 0x0404 → 0x0808 → ... → 점점 촘촘한 점선
        glLineStipple(factor, pattern);
        glBegin(GL_LINES);
        glVertex3f(-80.0f, y, 0.0f);  // 선의 시작점 (왼쪽)
        glVertex3f(80.0f, y, 0.0f);   // 선의 끝점 (오른쪽)
        glEnd();
        pattern = pattern * 16;
    }

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
