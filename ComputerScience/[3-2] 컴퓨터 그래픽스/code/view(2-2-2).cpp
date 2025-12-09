// 이 코드는 "고정된 직사각형(OpenGL glOrtho 고정 좌표계)로 인해 창 사이즈를 바꿔도 사각형 비율이 유지되지 않는 문제"를 보여주는 예제이다.
// 즉, glViewport와 glOrtho 설정을 고정값으로 해 정사각형이 창 크기 변화에 따라 찌그러지는 현상을 확인하는 문제용 예제 코드이다.

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

// 장면 렌더링
void RenderScene(void) {
    std::cout << "RenderScene" << std::endl;

    int SizeA = 200;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // glViewport와 glOrtho를 모두 고정한 상태.
    glViewport(0, 0, 680, 480);

    glOrtho(0, 640, 0, 480, -1, 1); // 실제 화면에 보이는 이미지 크기

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 현재 색상을 사용하여 화면을 지운다
    glClear(GL_COLOR_BUFFER_BIT);

    // 창 크기와 무관하게 고정된 좌표계에서 사각형을 그림
    glColor3f(1.0f, 0.0f, 0.0f);
    glRectf(0, 240, 320, 0);

    // 드로잉 명령을 전달한다. 명령을 강제로 실행하여 즉시 렌더링, GLUT
    glFlush();
}

// 랜더링 컨텍스트 설정
void SetupRc(void)
{
    std::cout << "SetupRc" << std::endl;
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(500, 500);
    glutCreateWindow("Fixed Square"); // 창의 제목도 "고정 사각형"으로 설명

    SetupRc();
    glutDisplayFunc(RenderScene);

    glutMainLoop();
    return 0;
}