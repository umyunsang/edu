// -----------------------------------------------------------------------------
// 이 예제는 OpenGL에서 glViewport와 glOrtho(투영 좌표계)를 "고정" 값으로 설정한 상태에서
// 창 크기를 변경해도 빨간 사각형의 비율(종횡비)이 창의 실제 비율과 맞지 않아 찌그러져 보이는 현상을 실험하는 코드입니다.
// 즉, 화면 좌표계와 실제 창 크기를 일치시키지 않고 고정해두면 사각형이 창 크기 변화에 따라 왜곡됨을 확인할 수 있습니다.
// glViewport(0,0,500,500), glOrtho(0,500,0,500,-1,1)로 "고정"되어 있음에 주의.
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

    int SizeA = 200;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 뷰포트와 투영을 고정값(500x500)에 맞춰 설정
    glViewport(0, 0, 500, 500);

    // 화면에 보이는 투영 좌표계도 0~500으로 "고정" 설정
    // 창 크기가 달라져도 고정 좌표계에 그려지므로 종횡비가 맞지 않게 찌그러짐.
    glOrtho(0, 500, 0, 500, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 파란색 배경으로 화면 전체를 지움
    glClear(GL_COLOR_BUFFER_BIT);

    // 빨간 사각형 그리기 (좌표: (250,250) ~ (375,375)), 화면 중심 기준 대략 125x125크기
    glColor3f(1.0f, 0.0f, 0.0f);
    glRectf(375, 250, 250, 375);

    // 그리기 명령 즉시 실행
    glFlush();
}

// 렌더링 초기화: 배경색을 파란색으로 설정
void SetupRc(void)
{
    std::cout << "SetupRc" << std::endl;
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    // 창의 실제 크기는 640x480으로 설정하지만,
    // glViewport와 glOrtho는 500x500으로 고정됨.
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(500, 500);
    glutCreateWindow("Fixed Square (종횡비 왜곡 예제)");

    SetupRc();
    glutDisplayFunc(RenderScene);

    glutMainLoop();
    return 0;
}