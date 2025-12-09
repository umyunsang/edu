#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

void RenderScene(void) {
    GLfloat x, y, z, angle;
    GLfloat pointSize = 1.0f;
    GLfloat r = 1.0f, g = 0.0f, b = 0.0f;  // 시작 색상: 빨강

    glClear(GL_COLOR_BUFFER_BIT);

    glPushMatrix();

    glRotatef(45, 1.0f, 0.0f, 0.0f);
    glRotatef(45, 0.0f, 1.0f, 0.0f);

    z = -50.0f;

    for(angle = 0.0f; angle <= (2.0f * 3.14f) * 3.0f; angle += 0.1f) {
        x = 50.0f * cos(angle);
        y = 50.0f * sin(angle);

        glColor3f(r, g, b);
        r -= 0.01f;
        g += 0.01f;
        b += 0.01f;

        glPointSize(pointSize);
        pointSize += 0.05f;

        glBegin(GL_POINTS);
        glVertex3f(x, y, z);
        glEnd();

        z += 0.5f;
    }
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
    glutCreateWindow("점 크기가 증가하며 색상이 변하는 나선형");
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // 시스템이 지원하는 점 크기 범위를 확인
    GLfloat sizes[2];  // sizes[0]: 최소 크기, sizes[1]: 최대 크기
    GLfloat step;      // 점 크기 변경 시 최소 단위
    glGetFloatv(GL_POINT_SIZE_RANGE, sizes);           // 지원 가능한 점 크기 범위 가져오기
    glGetFloatv(GL_POINT_SIZE_GRANULARITY, &step);     // 점 크기 증가 최소 단위 가져오기
    std::cout << "점 크기 범위: " << sizes[0] << " ~ " << sizes[1] << std::endl;
    std::cout << "점 크기 증가 단위: " << step << std::endl;

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glutMainLoop();

    return 0;
}
