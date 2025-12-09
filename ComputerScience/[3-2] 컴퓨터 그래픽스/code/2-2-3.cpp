// ----------------------------
// 이 예제는 OpenGL에서 창 크기(윈도우 크기)를 바꿀 때
// 화면에 그려지는 사각형(정사각형)의 비율이 어떻게 달라지는지,
// 즉, glOrtho와 glViewport 설정만으로는 비율이 왜 일치하지 않는지,
// 좌표계와 실제 픽셀 화면의 종횡비 불일치 현상을 실험해보는 예제입니다.
// 창의 크기를 마우스로 조절하면 빨간 정사각형이 찌그러지는 현상을 관찰할 수 있습니다.
// ----------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>

// 사각형(정사각형)을 한 쌍 그려서 비율 변화 관찰
void RenderScene(void) {
    // 화면을 파란색(배경색)으로 지우고 시작
    glClear(GL_COLOR_BUFFER_BIT);

    // 빨간색 정사각형 하나 그리기 (좌표계 원점 기준)
    glColor3f(1.0f, 0.0f, 0.0f);
    glRectf(-50.0f, -50.0f, 50.0f, 50.0f);

    glFlush();
}

// 창 크기(Reshape) 이벤트 콜백
// 창의 폭(w)과 높이(h)로부터 glViewport와 glOrtho를 재조정
// 단순히 종횡비(aspect ratio)에 맞춰서 glOrtho를 자동으로 구성,
// 이 과정에서 실제 정사각형이 흰화면 픽셀 비율과 달라지기 때문에
// 찌그러진 정사각형이 나타남을 확인할 수 있음
void ChangeSize(GLsizei w, GLsizei h) {
    GLint wSize = 100;    // 가상의 논리 좌표계 절반 길이
    GLfloat aspectRatio;

    // 높이가 0이 되는 것을 막기 위해 1로 고정
    if (h == 0) {
        h = 1;
    }
    // 뷰포트는 전체 창 크기로 설정
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    aspectRatio = (GLfloat)w / (GLfloat)h; // 창의 종횡비 계산

    // glOrtho 설정(좌표계 기준)을 창 종횡비에 따라 자동 조정
    // 실제 픽셀/좌표계 비율이 불일치하기 때문에 사각형이 찌그러져 보임
    if (w <= h) {
        glOrtho(-wSize, wSize, -wSize / aspectRatio, wSize / aspectRatio, -1, 1);
    } else {
        glOrtho(-wSize * aspectRatio, wSize * aspectRatio, -wSize, wSize, -1, 1);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int main(int argc, char** argv) {
    // GLUT 초기화 및 윈도우 생성
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(400, 400);
    glutCreateWindow("simple - aspect ratio (종횡비 실험)");

    // 콜백 함수 등록
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);

    glutMainLoop();
    return 0;
}