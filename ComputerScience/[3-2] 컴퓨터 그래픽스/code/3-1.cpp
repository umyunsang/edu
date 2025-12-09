// ----------------------------
// 이 예제는 OpenGL에서 두 빨간 점 [(0, 0, 0), (50, 50, 50)]을 glBegin~glEnd 블록 내에서 한 번에 그리는 예제입니다.
// (참고: (0,0,0) 점이 안보이면, 투영/좌표계 또는 z 값 범위를 확인하세요. 아래에서는 z=0 평면과 z=50 평면에 점을 찍습니다.)
// "50,50,50" 점이 보이는데 "0,0,0" 점이 안보인다면, glOrtho의 zNear, zFar 값을 더 넓게 잡아야 합니다.
// ----------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0f, 0.0f, 0.0f); // 빨간색으로 설정
    glPointSize(8.0f); // 점이 더 잘 보이도록 크기 증가

    glBegin(GL_POINTS);
        glVertex3f(0.0f, 0.0f, 0.0f);          // (0,0,0) 점
        glVertex3f(50.0f, 50.0f, 50.0f);       // (50,50,50) 점
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

    // zNear/zFar 값을 (예: -100~+100)처럼 좌표 범위를 충분히 넓게 잡으면 두 점이 모두 보인다.
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
    glutCreateWindow("Point Drawing Example (점 그리기 예제)");

    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);

    glutMainLoop();
    return 0;
}