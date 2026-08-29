## --- [Page 1] ---
분산처리

1

Computer & Ai

Department of Computer Engineering

Parallel Computing Stanford CS149, Fall 2024 수업 자료 참고

GPU Architecture &
CUDA Programming

## --- [Page 2] ---
Review: how to run code on a CPU

History: 원래 3D 게임 가속을 위해 설계된 그래픽 프로세서가 다음과 같은 광범위한 애
플리케이션을 위한 고도의 병렬 컴퓨팅 엔진으로 진화한 과정을 살펴보자:

-deep learning

- computer vision

- scientific computing

▪CUDA 언어를 사용한 GPU 프로그래밍

▪GPU 아키텍처에 대해 자세히 살펴보기

## --- [Page 3] ---
basic GPU architecture 이해

멀티코어 칩

•
단일 코어 내 SIMD 실행(동일한 명령을 수행하는 여러 실행 유닛)

•
단일 코어에서 멀티스레드 실행(하나의 코어에서 여러 스레드가 동시에 실행)

Memory
DDR5 DRAM

(a few GB)

~150-300 G
B/sec
(high end GPUs)

GPU

## --- [Page 4] ---
basic GPU architecture 이해

## --- [Page 5] ---
GPU가 본연의 설계된 기능: 3D 렌더링

Image credit: Henrik Wann Jensen

Input: description of a scene
(장면을 만들기 위한 정보):
3D surface geometry (e.g., triangle mesh)
surface materials, lights, camera, etc.

Output: image of the scene

렌더링 작업의 간단한 정의: 3D 메시의 각 트라이앵글이 이미지의 각 
픽셀 모양에 어떻게 기여하는지 계산하는 것?

https://threejs.org/examples/?q=tea#webgl_geometry_teapot

## --- [Page 6] ---
GPU가 본연의 설계된 기능: 3D 렌더링

Unreal Engine Kite Demo (Epic Games 2015)

Real-time (30 fps) on a high-end GPU

## --- [Page 7] ---
Real-time graphics primitives (entities)

Vertices
(points in space)

Primitives
(e.g., triangles, points, lines)

1

2

3

4

표면을 3D 삼각형 메시로 표현

three.js examples (threejs.org)

## --- [Page 8] ---
Real-time graphics primitives (entities)

Vertices
(points in space)

Primitives
(e.g., triangles, points, lines)

1

2

3

4

Pixels (in an image)
Fragments

## --- [Page 9] ---
Transformations and the Graphics Pipeline

## --- [Page 10] ---
Transformations and the Graphics Pipeline

## --- [Page 11] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | Input vertex buffer 입력: 3D 공간의 정점 목록(및 프리미티브에 대한 연결성) Vertex Generation 3D vertex str 예: 정점 3개마다 삼각형을 정의. eam list of positions = { _ _ v0x, v0y, v0z, v1x, v1y, v1x, triangle 0 = {v0, v1, v2} v2x, v2y, v2z, triangle 1 = {v1, v2, v3} v3x, v3y, v3x }; |

## --- [Page 12] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | 1단계: 씬 카메라 위치가 주어지면 화면 에서 정점이 어디에 있는지 계산. Input vertex buffer v2 Vertex Generation v3 v1 3D vertex strea m v0 Vertex Processing Projected vertex stream v2 v3 v1 v0 |

## --- [Page 13] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | Input vertex buffer 2단계: 정점을 프리미티브로 그룹화하기 Vertex Generation 3D vertex strea m v2 Vertex Processing v3 v1 Projected vertex stream v0 Primitive Generation Primitive stream t0 t1 |

## --- [Page 14] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | 3단계: 프리미티브가 겹치는 각 픽셀에 대해 Input vertex buffer 하나의 조각을 생성. Vertex Generation 3D vertex strea t0 t1 m Vertex Processing Projected vertex stream Primitive Generation Primitive stream Fragment Generation (“Rasterization”) Fragment stream |

## --- [Page 15] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | 4단계: 각 조각에 대한 프리미티브 색상 계산(씬 라 Input vertex buffer 이팅 및 프리미티브 머티리얼 프로퍼티 기반) Vertex Generation 3D vertex strea m Vertex Processing Projected vertex stream Primitive Generation Primitive stream Fragment Generation (“Rasterization”) Fragment stream Fragment Processing Colored fragment stream |

## --- [Page 16] ---


|  |  |
| --- | --- |
|  | Rendering a picture |
|  | Input vertex buffer 5단계: 출력 이미지에서 카메라에 "가장 가까운 Vertex Generation 조각"의 색상을 넣음. 3D vertex strea m Vertex Processing Projected vertex stream Primitive Generation Primitive stream Fragment Generation (“Rasterization”) Fragment stream Fragment Processing Colored fragment stream Output image buffer (pixels) Pixel Operations |

## --- [Page 17] ---


|  |  |
| --- | --- |
|  | Real-time graphics pipeline |
|  | Input vertex buffer Vertex Generation 3D vertex strea m • 그림을 정점, 프리미티브, 조각, 픽셀에 대한 일련의 연산 Vertex Processing 으로 렌더링하는 과정을 추상화 함. Projected vertex stream Primitive Generation Primitive stream Fragment Generation (“Rasterization”) Fragment stream Fragment Processing Colored fragment stream Output image buffer (pixels) Pixel Operations |

## --- [Page 18] ---


|  |  |
| --- | --- |
|  | 프래그먼트 처리 계산은 real-world material의 빛 반사를 시뮬레이션함. |
|  | Example materials: Images from Matusik et al. SIGGRAPH 2003 |

## --- [Page 19] ---


|  |  |
| --- | --- |
|  | 초기 그래픽 프로그래밍(OpenGL API) |
|  | • 그래픽스 프로그래밍 API는 프로그래머가 씬 조명(Light)및 머티리얼(Material)의 파라미터를 설정 할 수 있는 메커니즘을 제공. • glLight(light id, parameter id, parameter value) _ _ _ • Examples of light parameters: color, position, direction • glMaterial(face, parameter id, parameter value) _ _ • Examples of material parameters: color, shininess |

## --- [Page 20] ---


|  |  |
| --- | --- |
|  | 세상의 다양한 소재와 조명! |
|  |  |

## --- [Page 21] ---


|  |  |
| --- | --- |
|  | 세상의 다양한 소재와 조명! |
|  | Input vertex buffer Graphics shading languages Vertex Generation 3D vertex stream • 머티리얼과 조명을 프로그래밍 방식으로 지정하여 Vertex Processing 그래픽 파이프라인의 기능을 확장할 수 있음! Projected vertex st ream ➢다양한 머티리얼 지원 ➢다양한 조명 조건 지원 Primitive Generation Primitive stream • 프로그래머는 특정 단계에 대한 파이프라인 로직 을 정의하는 미니 프로그램(“셰이더”)을 제공. Fragment Generation (“Rasterization”) • 파이프라인은 셰이더 기능을 입력 스트림의 모든 Fragment stream 요소에 매핑. Fragment Processing Colored fragment st ream Output image b uffer (pixels) Pixel Operations |

## --- [Page 22] ---


|  |  |
| --- | --- |
|  | Example fragment shader program * |
|  | Run once per fragment (per pixel covered by a triangle) myTexture is a texture map 다음 OpenGL shading 언어(GLSL) 셰이더 프로 그램: Fragment processing단계의 동작을 정의. uniform sampler2D myTexture; read-only global variables uniform float3 lightDir; varying vec3 norm; per-fragment inputs varying vec2 uv; void myFragmentShader() { vec3 kd = texture2D(myTexture, uv); “fragment shader” kd *= clamp(dot(lightDir, norm), 0.0, 1.0); (a.k.a kernel function mapped return vec4(kd, 1.0); onto input fragment stream) } per-fragment output: RGBA surface color at pixel * Syntax/details of this code not important to 15-418. What is important is that it’s a kernel function operating on a stream of inputs. |

## --- [Page 23] ---


|  |  |
| --- | --- |
|  | 음영 처리된 결과 |
|  | 이미지에는 표면이 덮은 각 픽셀에 대한 myFragmentShader의 출력이 포함(여러 표면이 덮은 픽셀은 카메라에 가장 가까운 표면의 출력을 포함). |

## --- [Page 24] ---


|  |  |
| --- | --- |
|  | 음영 처리된 결과 |
|  | 음영 처리된 결과 이미지에는 표면이 덮은 각 픽셀에 대한 myFragmentShader의 출력이 포함(여러 표면이 덮은 픽셀은 카메라에 가장 가까운 표면의 출력을 포함). |

## --- [Page 25] ---


|  |  |
| --- | --- |
|  | 셰이더를 다른 연산에 사용할 수 없나요? |
|  | • OpenGL 출력 이미지 크기를 출력 배열 크기(예: 512 x 512)로 설정. • 화면을 정확히 덮는 2개의 삼각형을 렌더링 (픽셀당 하나의 셰이더 계산 = 하나의 셰이더 계산 출력 이미지 요소). • 이제 GPU를 데이터 병렬 프로그래밍 시스템처럼 사용할 수 있다. • Fragment shader function는512 x 512 요소 컬렉션에 매핑됩니다. |

## --- [Page 26] ---


|  |  |
| --- | --- |
|  | “GPGPU” 2002-2003 |
|  | GPGPU = “general purpose” computation on GPUs Coupled Map Lattice Simulation [Harris 02] Sparse Matrix Solvers [Bolz 03] Ray Tracing on Programmable Graphics Hardware [Purcell 02] |

## --- [Page 27] ---
2001-2003년경 관측

이러한 GPU는 대규모 데이터 모음(버텍스, 조각, 픽셀 스트림)에서 동일한 
연산(셰이더 프로그램)을 수행하는 데 매우 빠른 프로세서입니다.

2001년: 프로그래밍 가능한 
셰이딩이 가능한 최초의 단일 칩 
GPU인 Nvidia GeForce 3 출시

하지만 많은 제약(루프 없음)

2002: ATI Radeon 9700, 루프 및 FP 
수학을 갖춘 셰이더 지원

잠깐만요! 저한테는 데이터 병렬 
처리와 비슷하게 들리는데요!  
90년대 이국적인 슈퍼컴퓨터의 
데이터 병렬 처리를 기억합니다.

## --- [Page 28] ---
• Stanford graphics lab research project

• Abstract GPU hardware as data-parallel processor

▪
Brook 컴파일러는 일반 스트림 프로그램을 당시의 GPU에서 실행할 수 있는 그
래픽 명령어(예: drawTriangles)와 그래픽 셰이더 프로그램 세트로 변환함

[Buck 2004]

Brook stream programming language (2004)

## --- [Page 29] ---
CUDA Programming Language

## --- [Page 30] ---
GPU compute mode

## --- [Page 31] ---
Review: how to run code on a CPU

⚫사용자가 멀티코어 CPU에서 프로그램을 실행하려고

한다고 가정해 보자

✓OS가 프로그램 텍스트를 메모리에 로드.
✓OS가 CPU 실행 컨텍스트 선택
✓OS가 프로세서를 중단하고 실행 컨텍스트를 준비

(레지스터, 프로그램 카운터 등의 내용을 설정하여 실행 컨텍스트 준비).
✓실행!
✓프로세서는 실행 컨텍스트에서 유지된 환경 내에서

명령어 실행을 시작.

## --- [Page 32] ---
How to run code on a GPU (prior to 2007)

⚫사용자가 GPU를 사용하여 그림을 그리려고 한다고

가정해 보자..

✓애플리케이션(그래픽 드라이버를 통해)이 GPU 셰이더 프로

그램 바이너리를 제공.

✓애플리케이션이 그래픽 파이프라인 파라미터를 설정.

(예: 출력 이미지 크기)

✓애플리케이션이 GPU에 버텍스 버퍼를 제공.

✓애플리케이션이 GPU에 “그리기” 명령을 보냄:

✓drawPrimitives(vertex_buffer)

•
이것은 GPU 하드웨어에 대한 유일한 인터페이스였음.
•
GPU 하드웨어는 그래픽 파이프라인 계산만 실행할 수 있었음.

## --- [Page 33] ---
NVIDIA Tesla architecture (2007)

에서 데이터를 복사.

②애플리케이션(그래픽 드라이버를 통해)이 GPU에 단

일 커널 프로그램 바이너리 제공

③애플리케이션이 GPU에 SPMD 방식으로 커널을 실행

하도록 지시함.

(“이 커널의 N 인스턴스 실행”)

launch(myKernel, N)

흥미롭게도 이것은 그래픽 연산 drawPrimitives()보다 훨씬 
간단한 연산입니다.


| 비그래픽 |  | 전용 | (“ | 컴퓨팅 |  | 모드 | ” | ) | 인터페이스 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


|  |  |  | GPU | 의 |  | 프로그래머블 |  |  |  |  |  | 코어에서 |  | 비그래픽 |  | 프로 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 그램을 |  | 실행하고자 |  |  |  |  |  | 한다고 |  | 가정 |  |  |  |  |  |  |

## --- [Page 34] ---


|  |  |
| --- | --- |
|  | CUDA programming language |
|  | ⚫ 2007년에 엔비디아 테슬라 아키텍처와 함께 도입. ⚫ GPU에서 실행되는 프로그램을 표현하는 "C와 유사한" 언어. ⚫ 비교적 낮은 수준: CUDA의 추상화는 최신 GPU의 기능/성능 특성과 거의 일치. (설계 목표: 낮은 추상화 거리 유지). ✓ 참고: OpenCL은 CUDA의 개방형 표준 버전. ✓ CUDA는 NVIDIA GPU에서만 실행. ✓ OpenCL은 여러 공급업체의 CPU 및 GPU에서 실행. ✓ CUDA에 대해 말하는 거의 모든 내용은 OpenCL에도 적용됨 |

## --- [Page 35] ---


|  |  |
| --- | --- |
|  | CUDA programming language |
|  | 1.CUDA 프로그래밍 추상화. 2.최신 GPU에서 CUDA 구현. 3.GPU 아키텍처에 대한 자세한 내용. • 이 강의를 통해 고려해야 할 사항 ✓CUDA는 데이터 병렬 프로그래밍 모델인가요? ✓CUDA는 공유 주소 공간 모델의 예시인가요? ✓아니면 메시지 전달 모델의 예시일까요? ✓ISPC 인스턴스와 작업을 비유할 수 있을까요? pthread는 어떤가요? |

## --- [Page 36] ---


|  |  |
| --- | --- |
|  | 설명 (다시 시작합니다) |
|  | • CUDA 용어를 사용하여 CUDA 추상화를 설명 • 특히 CUDA 스레드라는 용어를 사용할 때 주의. ➢ CUDA 스레드는 논리적 제어 스레드에 해당한다는 점에서 pthread와 유사한 추상 화를 제공하지만 CUDA 스레드의 구현은 매우 다름. |

## --- [Page 37] ---
• 스레드 ID는 최대 3차원까지 가능(아래 2D 예시).

• 다차원 스레드 ID는 자연적으로 N-D인 문제에 편리함.

CUDA 프로그램은 concurrent threads의 계층 구조로 구성

## --- [Page 38] ---
CUDA blocks map to GPU cores (streaming multiprocessors)

## --- [Page 39] ---
• gridDim: 그리드의 차원
• blockIdx: 그리드 내 block index
• blockDim: block의차원
• threadIdx: block내 thread index

Grid, Block, and Thread

## --- [Page 40] ---
각 스레드는 블록 내 자신의 위치(threadIdx)와 그리드 내 블록의 
위치(blockIdx)에서 전체 그리드 스레드 ID를 계산(유일성)

많은 CUDA 스레드의 대량 실행 
“CUDA 스레드 블록 그리드 실행”
모든 스레드가 종료되면 호출이 반환.

“호스트” 코드 :
CPU에서 일반 C/C++ 애플리케이션의 
일부로 실행되는직렬 실행

SPMD execution of device kernel function:

“CUDA device” code: kernel function (__global__ denotes a CUDA kernel 
function) runs on GPU

Basic CUDA syntax

## --- [Page 41] ---
Basic CUDA syntax

#include <stdio.h>

__device__ void device_strcpy(char *dst, const char *src) {
    while (*dst++ = *src++);
}

__global__ void kernel(char *A) {
   device_strcpy(A, "Hello, World!");
}

int main() {
   char *d_hello;
   char hello[32];

cudaMalloc((void**)&d_hello, 32);

kernel<<<1,1>>>(d_hello);

cudaMemcpy(hello, d_hello, 32, cudaMemcpyDeviceToHost);

cudaFree(d_hello);

puts(hello);
}

## --- [Page 42] ---
Basic CUDA syntax

## --- [Page 43] ---
실행을 호스트 코드와 디바이스 코드로 분리하는 것은 프로그래머가 정적으로 수행.

호스트와 디바이스 코드의 명확한 분리

## --- [Page 44] ---
#include <stdio.h>

__device__ void device_strcpy(char *dst, const char *src) {
    while (*dst++ = *src++);
}

__global__ void kernel(char *A) {
    device_strcpy(A, "Hello, World!");
}

int main() {
   char *d_hello;
   char hello[32];

cudaMalloc((void**)&d_hello, 32);

kernel<<<1,1>>>(d_hello);

cudaMemcpy(hello, d_hello, 32, cudaMemcpyDeviceToHost);

cudaFree(d_hello);

puts(hello);
}

Basic CUDA syntax
호스트와 디바이스 코드의 명확한 분리

## --- [Page 45] ---
실행을 호스트 코드와 디바이스 코드로 분리하는 것은 프로그래머가 정적으로 수행.

호스트와 디바이스 코드의 명확한 분리

## --- [Page 46] ---
• 커널 호출 횟수는 데이터 수집 크기에 따라 결정되지 않음.
• (kernel launch은 그래픽 셰이더 프로그래밍의 경우처럼 맵(커널, 컬렉션)이 아님).

SPMD 스레드 수는 프로그램에 명시

## --- [Page 47] ---
CUDA execution model

## --- [Page 48] ---
고유한 호스트 및 디바이스 주소 공간

CUDA memory model

## --- [Page 49] ---
Move data between address spaces

memcpy primitive

cudaMemcpy를 떠올리면 어떤 것이 떠오르나요?

## --- [Page 50] ---
고유한 호스트 및 디바이스 주소 공간

GPU 메모리 구조

CUDA memory model

## --- [Page 51] ---
Per-block
shared memory

Per-thread
private memory

Readable/ writable by all 
threads in block

Readable/ writable by t
hread

Device global

memory

Readable/writable by 
all threads

커널에 표시되는 세 가지 메모리 유형

CUDA device memory model

•
서로 다른 주소 공간은 프로그램에서 서로 다른 지역성을 반영, 이는 CUDA의 GPU 
구현 효율성에 중요한 영향을 미침
•
(예: 특정 스레드가 동일한 변수에 액세스한다는 것을 선험적으로 알고 있는 경우 
스레드를 어떻게 예약할 수 있을까요?.

## --- [Page 52] ---
CUDA device memory model

프로그래머가 메모리 계층구조를 직접 제어가능

## --- [Page 53] ---
CUDA device memory model 특징

◆Per-Thread (local Memory)

•Registers
➢On-chip memory
➢가장 빠르고, 가장 작은 메모리
➢보통 블록 1개가 8k-64k 32bit register를 사용한다.
➢각 SM 내부에 존재
➢thread들이 register를 나누어서 가져간다.(register의 영역을 나누어서 사용)

따라서 context switching이 필요 없다.

•Local Memory
➢Off-chip memory
➢SM 내부에 없고, DRAM 영역 내부에 존재
➢register에 다 올라가지 못하는 것들은 Local Memory에 올려서 동작
➢Register에 비해 느리지만 더 많은 공간을 활용할 수 있다.(크게 제약이 없다.)

## --- [Page 54] ---
CUDA device memory model 특징

◆Per-Block (shared Memory)

• Shared Memory(__shared__)
✓On-chip memory
✓SM 내부에 존재해 빠르지만 공간이 적다.
✓블록 안의 모든 thread들이 공유한다.(모든 thread가 접근이 가능하다.)
✓thread 간의 communication가능
✓SM 내부의 block들이 shared memory 공간을 나누어서 활용(block 개수 제한)

## --- [Page 55] ---
CUDA device memory model 특징

◆Per-Gird (global Memory)

•
Global Memory
✓모든 thread가 활용할 수 있다.
✓off-chip memory
✓가장 느리지만 가장 큰 공간을 가진다.
✓Host에서 접근이 가능하다.
•
Constant Memory
✓읽기 전용 메모리 : 읽을 수만 있다.
✓선언은 global로 하고 host를 통해 초기화 한다.
✓공간이 적지만 빠르다.
✓이 영역을 위한 캐시가 존재한다.(캐시 영역 지원)
✓기본적으로 DRAM에 저장되어 있지만 할당되어 있는 cache가 존재해서 자주 읽어오는 값들을 저장해 놓으면 빠르게 가져가 활용

할 수 있다.(인자 값과 같은 것들..)
•
Texture Memory
✓Constant memory와 비슷
✓2차원 array를 다루는 것에 최적화
✓Hardware 필터링 제공
✓기본적으로 graphic에 활용되므로 cuda에서 많이 사용하지는 않는다.

## --- [Page 56] ---
GPU Caches (Hardware Manage)

◆L1 cache

• SM마다 존재한다.

• Shared memory와 같은 공간에 존재

• Shared memory 공간으로 활용하기도 한다.

• 따라서 L1 cache를 더 사용할 수도 사용하지 않을 수도 있다.

◆L2 cache

• SM 내부에 있는 것이 아니라 바깥쪽에 존재하는 캐시

• Shared by all SM

• Read-only constant and texture caches

## --- [Page 57] ---
◆On-chip memory(In each SM)

✓
Registers

✓
Shared Memory

✓
Constant cache, Texture cache

◆Off -chip memory

✓
Local memory

✓
Global memory

✓
Constant memory and Texture memory

✓
L2 cache

Physical Location of CUDA Memories

## --- [Page 58] ---
Memory Model 성능에 대한 특징

⚫일단 각각 Memory Model을 잘 이해하고 적당한 공간에 가져다가 사용해야 한다.

(Design your kernel and block)
⚫무작정 많은 스레드를 사용한다고 성능이 좋아지는 것은 아니다.
⚫결론적으로 Activie Wrap과 Activie Block이 많아야 성능이 좋아질 수 있다.

◆Activie Wrap

• 일부 thread가 요구하는 register 공간 할당받지 못할 수 있다.
✓(#thread in a block) * (register per thread) > registers in a SM
• Active warp
✓wrap 내부의 모든 스레드들이 요구하는 register 공간을 가져야 한다.
• Activie warp을 늘리기 위해 적당한 register 공간을 배분해야 한다.

◆Activie Block

•하나의 블록에서 메모리 자원이 요구되는 것들을 모두 가지고 있는 것(?) 
✓Register - all warps in a block are active wrap
✓Shared memory space
•SM 내의 Active blocks은 동시에 실행될 수 있다.
•Activie Block을 늘리기 위해 적당한 Shared memory 공간을 배분해야 한다.

## --- [Page 59] ---
Occupancy(얼마나 가득 차 있는가)

◆= (# of active warp) / (# of maximum warps)

◆이론적으로 가능한 warp에 비해 active warp이 얼마나 존재하는가

◆많은 active block이 있을수록 높은 성능의 병렬 처리가 될 수 있는 확률이 높다.

◆kernal과 thread layout을 최대한 많은 active warp을 낼 수 있도록 설계해야 한다.

✓
thread당 register의 수

✓
block당 thread의 수

✓
block당 shared memory 크기

## --- [Page 60] ---
CUDA example: 1D convolution

## --- [Page 61] ---
1D convolution in CUDA (version 1)
출력 요소당 하나의 스레드

## --- [Page 62] ---
1D convolution in CUDA (version 2)
One thread per output element: 메모리 계층별 input data in per-block shared memory

모든 스레드가 블록의 
“support 영역”을 글로벌 
메모리에서 공유 메모리로 
협력적으로 로드.

## --- [Page 63] ---
CUDA synchronization constructs

⚫__syncthreads()

➢Barrier: 블록의 모든 스레드가 이 지점에 도착할 때까지 대기

⚫Atomic operations

➢e.g.,  float atomicAdd(float* addr, float amount)

➢전역 메모리와 공유 메모리 변수 모두에 대한 원자 연산

⚫Host/device synchronization

➢커널 반환 시 모든 스레드에 대한 암시적 배리어(barrier)

## --- [Page 64] ---
Summary: CUDA abstractions

▪Execution: thread hierarchy

- 많은 스레드의 일괄 실행(정확하지 않은 표현.. 나중에 설명)
- 2단계 계층 구조: 스레드가 스레드 블록으로 그룹화됨

▪Distributed address space

- 호스트와 디바이스 주소 공간 간에 복사할 수 있는 Built-in memcpy
primitives
- 세 가지 유형의 디바이스 주소 공간
- 스레드당, 블록당(‘공유’) 또는 프로그램당(‘전역’).

▪스레드 블록의 스레드에 대한 Barrier synchronization primitive.
▪추가 동기화를 위한 원자 프리미티브(공유 및 전역 변수)

## --- [Page 65] ---
CUDA semantics

1백만 개 이상의 CUDA 스레드(8K 스레드 블록 이상)를 실행.

이 CUDA 프로그램을 실행하면 로컬 변수/스택의 인스턴스
가 1백만 개 생성되나요?

공유 변수의 8K 인스턴스(지원)

pthread_create() 또는 std::thread() 호출을 
구현하는 것을 고려:

스레드 상태를 할당:

- 스레드를 위한 스택 공간
- OS가 스레드를 예약할 수 있도록 제어 블록을

할당.

## --- [Page 66] ---
High-end GPU

(16+ cores)

Mid-range GPU

(6 cores)

Assigning work

•
CUDA 프로그램이 수정 없이 이 모든 GPU에서 실행되는 
것이 바람직함
•
참고: 제가 보여드린 CUDA 프로그램에는 num_cores라는 
개념이 없음.
•
(CUDA 스레드 실행은 데이터 병렬 모델 예제에서 forall
loop와 비슷한 개념.)

## --- [Page 67] ---
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {

__syncthreads();

launch 8K thread blocks

A compiled CUDA device binary includes:

• Program text (instructions)
• Information about required resources:
- 128 threads per block
- B bytes of local data per thread
- 128+2개=130 floats (520 bytes) of shared spac

e per thread block

int N = 1024 * 1024;
cudaMalloc(&devInput, N+2);  // allocate array in device memory
cudaMalloc(&devOutput, N);   // allocate array in device memory

// property initialize contents of devInput here ...

convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput);

CUDA compilation


|  | int index | = | blockIdx.x |  | * | blockDim.x |  | + | threadIdx.x | ; // thread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


|  | support[ |  |  | threadIdx.x |  |  |  | ] |  | = input[index]; |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | if ( |  | threadIdx.x |  |  |  | < 2 | ) { |  |  |  |  |  |
|  |  | support[ |  |  | THREADS PER BLK+threadIdx.x _ _ |  |  |  |  |  | ] = input[ | index+THREADS PER BLK _ _ | ]; |
|  | } |  |  |  |  |  |  |  |  |  |  |  |  |

|  | float result = 0.0f; // thread |  |  |  |  |  |  |  | - | local variable |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | for (int |  | i | =0; | i | <3; | i | ++) |  |  |  |  |
|  |  | result += support[ |  |  |  |  |  | threadIdx.x |  | + | i | ]; |

|  | output[index] = result; |
| --- | --- |


## --- [Page 68] ---
Thread block scheduler

Shared mem
Shared mem
Shared mem
Shared mem

Device global memory
(DRAM)

Kernel launch command from host
launch(blockDim, convolve)

. . .

Grid of 8K(8000) convolve thread blocks (커널 실행 시 지정)

블록 리소스(Block resource) 요구 사항:
(컴파일된 커널 바이너리에 포함) 
• 128 스레드
• 520바이트의 공유 메모리
• (128 x B) 바이트의 로컬 메모리

CUDA의 주요 가정: thread blocks 실행은 어떤 순서로든 수행될 수 
있음(블록 간 종속성 없음).

GPU 구현은 리소스 요구 사항을 존중하는 동적 스케줄링 정책을 사
용하여 thread blocks ("작업")을 코어에 매핑.

Shared mem is fast 
on-chip memory

Special HW

in GPU

CUDA thread-block assignment

## --- [Page 69] ---
Dynamic Scheduling 예제

## --- [Page 70] ---
기타 예시 
• ISPC의 작업 시작 구현
✓CPU에서 각 하이퍼 스레드마다 하나의 pthread를 생성.  나머지 프로그램 동안 스레드가 유지됨
• 웹 서버의 스레드 풀
✓ 스레드 수는 미결 요청 수가 아닌 코어 수의 함수.
✓ 웹 서버 시작 시 생성된 스레드는 작업이 도착할 때까지 대기.

일반적인 디자인 패턴의 또 다른 예시:worker “threads” POOL

Sub-problems
(일명“tasks”, “work”)


| 리소스 |  | 사전 |  | 할당 |
| --- | --- | --- | --- | --- |


## --- [Page 71] ---
NVIDIA V100 SM “sub-core”

## --- [Page 72] ---
NVIDIA V100 SM “sub-core”

## --- [Page 73] ---
NVIDIA V100 SM “sub-core”

동일한 “워프”의 32개 스레드에 대한

스칼라 레지스터

스레드 블록의 32개 스레드 그룹을 워프라고 함.

- 스레드 블록에서 스레드 0-31은 동일한 워프에 속함(스레드 32-63 등도

마찬가지)
-
따라서 256개의 CUDA 스레드가 있는 스레드 블록은 8개의 워프에 매핑.
- V100의 각 서브코어는 최대 16개의 워프 실행을 스케줄링하고 인터리빙할 수

있음.

## --- [Page 74] ---
NVIDIA V100 SM “sub-core”

워프의 스레드는 SIMD 방식으로 실행.

동일한 명령어를 공유하는 경우

- NVIDIA는 이를 SIMT(단일 명령어 다중 CUDA 스레드)라고 함.
- 32개의 CUDA 스레드가 동일한 명령어를 공유하지 않으면 서로 다른 실행으로 
인해 성능이 저하될 수 있음.
- 이 매핑은 ISPC가 프로그램 인스턴스를 Gang*으로 실행하는 방식과 유사

워프는 CUDA의 일부가 아니지만 최신 NVIDIA GPU에서 중요한 CUDA 구현 
세부 사항임.

동일한 “워프”의 32개

스레드에 대한 
스칼라 레지스터

* 하지만 GPU 하드웨어는 32개의 독립적인 CUDA 스레드가 하나의 명령을 공유하는지를 동적으로 확인하고, 이것이 
사실이면 32개의 스레드를 모두 SIMD 방식으로 실행함. 
CUDA 프로그램은 ISPC 갱처럼 SIMD 명령어로 컴파일되지 않음.

## --- [Page 75] ---
Instruction execution

워프에서 CUDA 스레드에 대한 명령 스트림...
(이 예제에서는 모든 명령이 독립적입니다.)

▪CUDA 스레드의 전체 워프가 이 명령어 스트림을 실행한다는 점을 기억하자.
▪따라서 각 인스트럭션은 워프의 32개 CUDA 스레드 모두에서 실행됨.
▪ALU가 16개이므로 전체 워프에 대한 인스트럭션을 실행하려면 두 개의 클럭이 필요함.

## --- [Page 76] ---
NVIDIA V100 GPU SM

This is one NVIDIA V100 streaming multi-processor (SM) unit

## --- [Page 77] ---
Running a thread block on a V100 SM

각 클럭마다 SM 코어 작동:

- 각 서브코어가 실행 가능한 하나의 워프(파티션의 16개 워프에서)를 선택
- 각 서브코어가 워프 내 CUDA 스레드의 다음 명령어를 실행(이 명령어는 발산에 따라 워프 내 CUDA

스레드의 전체 또는 하위 집합에 적용될 수 있음)

## --- [Page 78] ---
• Warp는 SM(Streaming Multi-processor)의 기본 실행 단위(unit of execution) 이다. 
• 스레드 블록의 그리드를 실행하면, 그리드의 스레드 블록들은 SM들로 분배. 
• 스레드 블록이 SM에 스케쥴링되면 스레드 블록의 스레드들은 warp로 파티셔닝됨. 
• 32개의 연속된 스레드들로 구성된 하나의 warp는SIMT(Single Instruction Multiple Thread) 방식으로 실행. 
• 즉, 모든 스레드는 동일한 명령어를 실행하고, 각 스레드는 할당된 private data에 대해 작업을 수행합니다.

• 스레드 블록은 1,2,3차원으로 구성될 수 있음. 하지만, 하드웨어 관점에서 살펴보면, 모든 스레드는 1차원으
로 정렬됨. 
• 각 스레드는 블록에서 unique ID를 가지고 있음. 1차원 스레드 블록에서, 스레드의 unique ID는 CUDA에 
내장된 변수인 threadIdx.x에 저장되고, 연속된 threadIdx.x를 가진 스레드들이 warp로 그룹화됨.

Warps and Thread Blocks

## --- [Page 79] ---
NVIDIA V100 GPU (80 SMs)

## --- [Page 80] ---
Summary: geometry of the V100 GPU

* mul-add counted as 2 flops:

## --- [Page 81] ---
Running a CUDA program on a GPU

## --- [Page 82] ---
최신  GPU 아키텍츠의 발전

## --- [Page 83] ---
Running the convolve kernel

convolve hernel’s execution requirements:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520 바이트의 공유 메모리를 할당.

배열 크기 N이 매우 커서 호스트 측 커널 실행 시 수천 개의 스레드 블록이 생성된다고 가정 보자

#define THREADS_PER_BLK 128

convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, input_array, output_array);

아래의 가상의 2코어 GPU에서 이 프로그램을 실행해 보자.
(참고: 가상의 코어는 앞서 강의에서 설명한 V100 SM 코어보다 실행 유닛 수가 적고, 더 적은 활성 워프를 지원하며, 공유 
메모리도 더 적습니다.)

## --- [Page 84] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

Step 1: 호스트(HOST:CPU측)가 CUDA 디바이스(GPU)에 명령("이 커널 실행")을 전송.

## --- [Page 85] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

Step 2: 스케줄러가 블록 0을 코어 0에 매핑
(128개의 스레드와 520바이트의 공유 스토리지에 대한 실행 컨텍스트 예약).

## --- [Page 86] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

3단계: 스케줄러가 사용 가능한 실행 컨텍스트에 블록을 계속 매핑(인터리브 매핑 표시)

## --- [Page 87] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

3단계: 스케줄러가 블록을 사용 가능한 실행 컨텍스트에 계속 매핑(인터리브 매핑 표시). 
코어에 두 개의 스레드 블록만 적합(세 번째 블록은 공유 스토리지가 부족하여 3 x 520바이트> 1.5KB로 맞지 않음).

## --- [Page 88] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

4단계: 코어 0에서 스레드 블록 0이 완료됨

## --- [Page 89] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

5단계: 블록 4가 코어 0에 예약됨(실행 컨텍스트 0-127에 매핑됨)

## --- [Page 90] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

6단계: 코어 0에서 스레드 블록 2 완료

## --- [Page 91] ---
Running the CUDA kernel

커널의 실행 요구 사항:

각 스레드 블록은 128개의 CUDA 스레드를 실행.
각 스레드 블록은 130 x sizeof(float) = 520바이트의 공유 메모리를 할당

7단계: 스레드 블록 5가 코어 0에 예약됨(실행 컨텍스트 128-255에 매핑됨)

## --- [Page 92] ---
고급 스케줄링 질문:

(다음 예제를 이해했다면 CUDA 프로그램이 GPU에서 어떻게 실행되는지 잘 이해하고 있으며,

지금까지 강좌에서 다룬 작업 스케줄링 문제도 잘 이해하고 있는 것입니다.)

## --- [Page 93] ---
검토: "워프"란 무엇인가요?

워프는 NVIDIA GPU의 CUDA 구현 세부 사항이다.
최신 NVIDIA 하드웨어에서는 스레드 블록의 32개 CUDA 스레드 그룹이 32폭 SIMD 실행을 사용하여 동시
에 실행.

Fetch/Decode

…

thread 0 ctx

thread 31 ctx

thread 32 ctx

thread 63 ctx

thread 64 ctx

…

…

thread 383 ctx

Warp 0 context

Warp 1 context

…

thread 352 ctx

Warp 11 context

이 가상의 NVIDIA GPU 예시에서:
➢코어는 12개의 워프에 대한 컨텍스트를 유지.
➢각 클럭을 실행할 하나의 워프를 선택.

## --- [Page 94] ---
검토: "워프"란 무엇인가요?

⚫워프는 NVIDIA GPU의 CUDA 구현 세부 사항.

⚫최신 NVIDIA 하드웨어에서 스레드 블록의 32개 CUDA 스레드 그룹은 32폭 SIMD 실행을 사용하여

동시에 실행.

➢이러한 32개의 논리적 CUDA 스레드는 하나의 명령 스트림을 공유하므로 분산 실행으로 인해 성능이 저하

될 수 있다.

⚫명령어 스트림을 공유하는 32개의 스레드 그룹을 워프라고 함.

➢스레드 블록에서 스레드 0-31은 동일한 워프에 속함(스레드 32-63 등도 마찬가지).

➢따라서 256개의 CUDA 스레드가 있는 스레드 블록은 8개의 워프에 매핑.

➢지난번에 설명한 GTX 980의 각 "SMM" 코어는 최대 64개의 워프 실행을 스케줄링하고 인터리빙할 수 있

음.

➢따라서 ＂SMM＂ 코어는 여러 CUDA 스레드 블록을 동시에 실행할 수 있음

## --- [Page 95] ---
CUDA 추상화 구현

⚫ 스레드 블록은 시스템에서 원하는 순서대로 예약할 수 있음

- 시스템은 블록 간에 종속성이 없다고 가정.

- 논리적으로는 동시적이다.

-
ISPC 작업과 매우 비슷?

⚫동일한 블록의 CUDA 스레드가 동시에 실행됨(동시에 라이브).

-
블록 실행이 시작되면 모든 스레드가 존재하고 레지스터 상태가 할당됨.

(이러한 의미는 시스템에 스케줄링 제약을 가함)

- CUDA 스레드 블록은 그 자체로 SPMD 프로그램(프로그램 인스턴스의 ISPC 갱과 같음)

- 스레드 블록의 스레드는 동시적이며 협력하는 “workers”임.

⚫CUDA 구현:

- NVIDIA GPU 워프는 ISPC instances gang과 유사한 성능 특성을 가짐(단, ISPC gang과 달리 워프 개념은 프로그래밍

모델*에 존재하지 않음).

- 스레드 블록의 모든 워프는 동일한 SM에 예약되므로 공유 메모리 변수를 통해 높은 BW/저지연 통신이 가능.

- 블록의 모든 스레드가 완료되면 블록 리소스(공유 메모리 할당, 워프 실행 컨텍스트)를 다음 블록에서 사용할 수 있게 됨.

## --- [Page 96] ---
히스토그램을 만드는 프로그램을 생각해 보자

⚫이 예제: 배열에 있는 값의 히스토그램을 작성합니다.

- 모든 CUDA 스레드는 전역 메모리의 공유 변수를 원자적으로 업데이트함.

⚫CUDA 스레드 블록이 독립성을 보장한다고 주장한 적이 없음. CUDA가 어떤 순서로든 스케줄링할 수 있는 권리를

보유한다고만 언급함.

⚫이것은 유효한 코드이다! 이러한 아토믹스 사용은 어떤 순서로든 블록을 스케줄링하는 구현의 기능에 영향을 미치지

않습다. (아토믹스는 상호 배제를 위해 사용되며, 그 이상은 아님).

## --- [Page 97] ---
Bonus slide: “persistent thread” CUDA 프로그래밍 스타일

•
아이디어: 기본 GPU 구현에서 지원하는 코어 수와 
코어당 블록 수에 대한 지식이 필요한 CUDA 코드를 
작성함.

•
프로그래머는 GPU를 채울 수 있는 만큼의 스레드 
블록을 정확히 실행함.

(프로그램은 GPU 구현에 대해 GPU가 실제로 모든 
블록을 동시에 실행할 것이라는 가정을 함. 윽!)

•
이제 블록에 대한 작업 할당은 전적으로 
애플리케이션에 의해 구현됨. (GPU의 스레드 블록 
스케줄러 우회)

•
이제 프로그래머의 정신 모델은 *모든* CUDA 
스레드가 한 번에 GPU에서 동시에 실행되고 있다는 
것임.

## --- [Page 98] ---
CUDA summary

• 실행 의미론(Execution semantics).
- 문제를 스레드 블록으로 분할하는 것은 데이터 병렬 모델의 정신에 따른 것 (머신 독립적이어야 함: 시스템이 
코어 수에 관계없이 블록을 스케줄링함).
- 스레드 블록의 스레드는 실제로 동시에 실행됨(협력하기 때문에 그래야만 함).
- 단일 스레드 블록 내부: SPMD 공유 주소 공간 프로그래밍.
- 이러한 실행 모델 간에는 미묘하지만 주목할 만한 차이점이 있음. 반드시 이해해야 함. (그리고 병렬 
프로그래밍 시스템을 접할 때마다 어떤 시맨틱이 사용되고 있는지 스스로에게 물어보세요).

• 메모리 시맨틱(Memory semantics)
- 분산 주소 공간: 호스트/장치 메모리.
-
디바이스 메모리 내에서 로컬/블록 공유/글로벌 변수를 스레딩함
- 로드/저장은 이들 간에 데이터를 이동함(따라서 로컬/공유/글로벌 메모리는 별개의 주소 공간으로 생각하는 
것이 정확함).

• 주요 구현 세부 사항(Key implementation details:):
- 스레드 블록의 스레드는 공유 메모리를 통해 빠르게 통신할 수 있도록 동일한 GPU “SM”에 스케줄링됨.
- 스레드 블록의 스레드는 GPU 하드웨어에서 SIMT 실행을 위해 워프로 그룹화됨.

## --- [Page 99] ---
One last point…

• 이 강의에서는 GPU의 프로그래머블 코어를 위한 CUDA 프로그램 작성에 대해 언급.

- 작업(CUDA 커널 실행으로 설명됨)은 하드웨어 작업 스케줄러를 통해 코어에 매핑.

• GPU 실행을 구동하기 위한 그래픽 파이프라인 인터페이스도 있습니다.

- 그리고 GPU의 흥미로운 비프로그래밍 기능은 대부분 그래픽 파이프라인 작업의 실행을 가속화하기

위해 존재.

- CUDA 프로그램을 실행할 때는 거의 “꺼져 있는” 상태.

• GPU가 그래픽 파이프라인을 효율적으로 구현하는 방법은 그래픽 클래스의 주제... *.