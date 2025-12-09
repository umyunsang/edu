// -----------------------------------------------------------------------------
// 이 예제는 OpenGL에서 glViewport와 glOrtho(좌표계)를 "고정" 값으로 설정해둔 상황에서
// 창의 크기를 바꿔도 빨간 사각형의 비율(종횡비)이 실제 창의 비율과 일치하지 않고 찌그러지는 현상을 실험해보는 예제입니다.
// 즉, 창 크기와 무관하게 좌표계를 (-320~320, -240~240)로 설정하고,
// glViewport도 고정(680x480)으로만 설정해서 정사각형이 창 크기 변화에 따라 실제 픽셀상에서 찌그러져 보임을 관찰합니다.
//
// 창의 크기를 수동으로 늘리거나 줄여봐도 사각형의 종횡비가 맞게 조정되지 않습니다.
// glViewport와 glOrtho를 "고정"하면 나타나는 비율 왜곡 현상을 확인하세요.
// -----------------------------------------------------------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

// 장면을 그리는 함수
void RenderScene(void) {
    std::cout << "RenderScene" << std::endl;

    // 좌표계/화면 비율 실험용 사각형(고정된 좌표계)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 고정된 viewport와 고정된 좌표계: 창 크기 변경과 무관하게 동작
    glViewport(0, 0, 680, 480);
    glOrtho(-320, 320, -240, 240, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 배경(파란색)으로 화면 지우기
    glClear(GL_COLOR_BUFFER_BIT);

    // 빨간 사각형 그리기 (좌측 기준)
    glColor3f(1.0f, 0.0f, 0.0f);
    glRectf(0, 240, 320, 0);

    // 명령 강제 실행
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
    glutInitWindowSize(640, 480); // "창" 자체의 초기 크기
    glutInitWindowPosition(500, 500);
    glutCreateWindow("Fixed Square (사각형 종횡비 고정오차 예제)");

    SetupRc();
    glutDisplayFunc(RenderScene);

    glutMainLoop();
    return 0;
}