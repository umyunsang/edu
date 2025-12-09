// ----------------------------
// 이 예제는 OpenGL에서 삼각함수를 이용하여 원 모양으로 점을 찍는 예제입니다.
// (참고: 모든 점이 보이지 않으면 glOrtho의 zNear, zFar 값을 충분히 넓게 잡아야 합니다.)
// ----------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <cmath>    // 삼각함수 사용을 위한 헤더

void RenderScene(void) {
    glfloat x,y,z, angle;
    glClear(GL_COLOR_BUFFER_BIT);  // 화면을 지움
    glColor3f(1.0f, 0.0f, 0.0f);   // 빨간색으로 설정

    // 회전 변환을 위해 현재 행렬을 스택에 저장
    glpushmatrix();

    // X축을 기준으로 45도 회전 (나선이 기울어진 효과)
    glrotatef(45, 1.0f, 0.0f, 0.0f);
    // Y축을 기준으로 45도 회전 (나선을 더 입체적으로 보이게 함)
    glrotatef(45, 0.0f, 1.0f, 0.0f);

    glpointsize(2.0f);      // 점 크기를 2픽셀로 설정
    glbegin(GL_POINTS);     // 점 그리기 시작

    z = -50.0f;             // 나선의 시작 z 좌표

    // angle을 0부터 6π(3바퀴)까지 0.1씩 증가시키며 반복
    for(angle = 0.0f; angle <= (2.0f * 3.14f) * 3.0f; angle += 0.1f) {
        // 원의 방정식: x = r*cos(θ), y = r*sin(θ)
        // 반지름 50인 원 위의 점을 계산
        x = 50.0f * cos(angle);
        y = 50.0f * sin(angle);

        glvertex3f(x, y, z);  // (x, y, z) 위치에 점을 찍음

        // z값을 0.5씩 증가시켜 나선형으로 만듦
        // (원을 그리면서 동시에 z축 방향으로 이동)
        z += 0.5f;
    }

    glend();                // 점 그리기 종료
    glpopmatrix();          // 저장했던 행렬을 복원
    glflush();              // 그리기 명령을 즉시 실행
}

// 초기 설정 함수 (이 예제에서는 실제로 사용되지 않음)
void SetupRc(void)
{
    std::cout << "SetupRc" << std::endl;
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);  // 배경색을 파란색으로 설정
}

// 창 크기가 변경될 때 호출되는 함수
void ChangeSize(GLsizei w, GLsizei h) {
    GLint wSize = 100;    // 가상의 논리 좌표계 절반 길이 (좌표 범위: -100~100)
    GLfloat aspectRatio;

    // 높이가 0이 되는 것을 방지 (0으로 나누기 방지)
    if (h == 0) {
        h = 1;
    }

    // 뷰포트를 창 전체 크기로 설정
    glViewport(0, 0, w, h);

    // 투영 행렬 모드로 전환
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();  // 단위 행렬로 초기화

    // 창의 종횡비 계산
    aspectRatio = (GLfloat)w / (GLfloat)h;

    // zNear/zFar 값을 넉넉하게 (-100 ~ +100)로 해서 모든 점이 보이게 함
    if (w <= h) {
        // 창이 세로로 긴 경우: x축은 고정, y축은 늘림
        glOrtho(-wSize, wSize, -wSize / aspectRatio, wSize / aspectRatio, -100, 100);
    } else {
        // 창이 가로로 긴 경우: y축은 고정, x축은 늘림
        glOrtho(-wSize * aspectRatio, wSize * aspectRatio, -wSize, wSize, -100, 100);
    }

    // 모델뷰 행렬 모드로 전환
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();  // 단위 행렬로 초기화
}

int main(int argc, char** argv) {
    // GLUT 라이브러리 초기화
    glutInit(&argc, argv);

    // 디스플레이 모드 설정: 단일 버퍼 + RGB 컬러 모드
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    // 창 크기를 500x500 픽셀로 설정
    glutInitWindowSize(500, 500);

    // 창의 초기 위치를 화면의 (400, 400)으로 설정
    glutInitWindowPosition(400, 400);

    // "나선형 점 그리기"라는 제목의 창 생성
    glutCreateWindow("나선형 점 그리기");

    // 배경색을 파란색으로 설정 (R=0, G=0, B=1, A=1)
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

    // 화면을 그릴 때 호출될 함수 등록
    glutDisplayFunc(RenderScene);

    // 창 크기가 변경될 때 호출될 함수 등록
    glutReshapeFunc(ChangeSize);

    // 이벤트 처리 루프 시작 (프로그램이 계속 실행됨)
    glutMainLoop();

    return 0;
}