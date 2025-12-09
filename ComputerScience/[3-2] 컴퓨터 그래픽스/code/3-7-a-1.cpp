// 3-7-a에서 그린 원을 z=10에 하나(blue, 지름 40) 그리고 z=20에 다른 하나(red, 지름 20)를 그림
// - 후면제거 고려해야 함, 반 시계방향으로 그림

// - glOrtho(-100, 100, -100, 100, -100, 100);로 고정: near<far
// - 카메라 위치는 기본 위치이고 방향도 기본 방향
// gluLookAt( 0.0f, 0.0f, 0.0f,  // 카메라 위치(eye)
//           0.0f, 0.0f, -1.0f,  // 바라보는 지점(center)
//           0.0f, 1.0f, 0.0f); // 카메라 상향 벡터(up)

// 화면에 나타나는 원은?
//
// ★ gluLookAt(0,0,0, 0,0,-1, 0,1,0)일 때: 두 원 모두 보임
// - 카메라가 원점에서 -z 방향을 바라봄
// - 원들은 z=10, z=20 (양수)에 위치
// - 후면 제거 비활성화로 양면 모두 렌더링됨
// - Blue(z=10)가 앞, Red(z=20)가 뒤에 있어서 둘 다 보임
//
// ★ gluLookAt(0,0,0, 0,0,1, 0,1,0)일 때: 파란색 원만 보임
// - 카메라가 원점에서 +z 방향을 바라봄
// - 후면 제거 비활성화로 양면 모두 렌더링됨
// - 하지만 Blue(z=10)가 Red(z=20)보다 카메라에서 더 멀리 있음
//   (이 시점에서는 z가 클수록 가까움)
// - Blue의 반지름(20)이 Red의 반지름(10)보다 2배 커서
//   Blue가 Red를 완전히 가림
// - 결과: 파란색 원만 보임
//
// ★★★ 중요: glOrtho에서 near > far인 경우 ★★★
// glOrtho(-100, 100, -100, 100, 100, -100)처럼 near > far로 설정하면:
// - 깊이 방향이 반전됨
// - 위에서 정리한 바라보는 방향의 의미가 또 한 번 반전됨
// - 예: gluLookAt(0,0,0, 0,0,-1, ...)일 때 실제로는 +z 방향을 보는 효과
// - 예: gluLookAt(0,0,0, 0,0,1, ...)일 때 실제로는 -z 방향을 보는 효과
// - 즉, near/far 순서가 바뀌면 깊이 판정이 뒤집혀서 결과가 반대가 됨

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_CULL_FACE);  // 후면 제거 비활성화 (양면 모두 렌더링)
    glEnable(GL_DEPTH_TEST);  // 깊이 테스트 활성화

    GLfloat angle, x, y;

    // z=20에 red 원 (지름 20, 반지름 10) 그리기
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(1.0f, 0.0f, 0.0f);  // 빨간색
        glVertex3f(0.0f, 0.0f, 20.0f);  // 중심점

        for (angle = 0.0f; angle <= 2.0f * PI + 0.01f; angle += (2.0f * PI) / 16.0f) {
            x = 10.0f * cos(angle);  // 반지름 10
            y = 10.0f * sin(angle);
            glVertex3f(x, y, 20.0f);
        }
    glEnd();

    // z=10에 blue 원 (지름 40, 반지름 20) 그리기
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(0.0f, 0.0f, 1.0f);  // 파란색
        glVertex3f(0.0f, 0.0f, 10.0f);  // 중심점

        for (angle = 0.0f; angle <= 2.0f * PI + 0.01f; angle += (2.0f * PI) / 16.0f) {
            x = 20.0f * cos(angle);  // 반지름 20
            y = 20.0f * sin(angle);
            glVertex3f(x, y, 10.0f);
        }
    glEnd();

    glutSwapBuffers();
}

void ChangeSize(GLsizei w, GLsizei h) {
    if (h == 0) {
        h = 1;
    }

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 직교 투영: near=-100, far=100
    glOrtho(-100, 100, -100, 100, -100, 100);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    gluLookAt(0.0f, 0.0f, 0.0f,    // 카메라 위치(eye)
              0.0f, 0.0f, -1.0f,   // 바라보는 지점(center)
              0.0f, 1.0f, 0.0f);   // 카메라 상향 벡터(up)
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(400, 100);
    glutCreateWindow("Depth Test - Two Circles");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);

    std::cout << "=========================================" << std::endl;
    std::cout << "깊이 테스트 (후면 제거 비활성화)" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Blue 원: z=10, 지름=40" << std::endl;
    std::cout << "Red 원: z=20, 지름=20" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "gluLookAt(0,0,0, 0,0,-1, ...): 두 원 모두 보임" << std::endl;
    std::cout << "gluLookAt(0,0,0, 0,0,1, ...): Blue 원만 보임" << std::endl;
    std::cout << "(Blue가 Red를 완전히 가림)" << std::endl;
    std::cout << "=========================================" << std::endl;

    glutMainLoop();

    return 0;
}
