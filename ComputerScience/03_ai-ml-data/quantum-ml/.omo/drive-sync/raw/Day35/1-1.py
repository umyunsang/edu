"""
Lab 01
간단한 TSP 문제 풀어보기 (Brute Force)

학습목표
----------------------------------
1. TSP 문제를 이해한다.
2. 모든 가능한 경로를 생성한다.
3. 각 경로의 Cost를 계산한다.
4. 가장 짧은 경로를 찾는다.
"""

from itertools import permutations


# =====================================================
# 도시 정보
# =====================================================

START_CITY = "A"
CITIES = ["B", "C", "D"]


# =====================================================
# 거리 정보
# =====================================================

distance = {
    ("A", "B"): 10,
    ("A", "C"): 15,
    ("A", "D"): 20,
    ("B", "C"): 35,
    ("B", "D"): 25,
    ("C", "D"): 30,
}


# =====================================================
# 거리 조회 함수
# =====================================================

def get_distance(city1, city2):
    """두 도시 사이의 거리를 반환"""

    if city1 == city2:
        return 0

    if (city1, city2) in distance:
        return distance[(city1, city2)]

    if (city2, city1) in distance:
        return distance[(city2, city1)]

    raise ValueError("거리 정보가 없습니다.")


# =====================================================
# 총 이동 거리 계산
# =====================================================

def calculate_cost(route):

    total = 0

    for i in range(len(route) - 1):
        total += get_distance(route[i], route[i + 1])

    return total


# =====================================================
# 가능한 모든 경로 생성
# =====================================================

def generate_routes():

    routes = []

    for order in permutations(CITIES):

        route = (START_CITY,) + order + (START_CITY,)

        routes.append(route)

    return routes


# =====================================================
# 결과 출력
# =====================================================

def print_result(routes):

    print("=" * 60)
    print("모든 가능한 경로")
    print("=" * 60)

    best_cost = float("inf")
    best_route = None

    for idx, route in enumerate(routes, start=1):

        cost = calculate_cost(route)

        print(
            f"{idx:2d}. "
            f"{' -> '.join(route):25s}"
            f" : {cost:3d} km"
        )

        if cost < best_cost:
            best_cost = cost
            best_route = route

    print()
    print("=" * 60)
    print("Optimization Result")
    print("=" * 60)
    print("Optimal Route")
    print(" -> ".join(best_route))
    print(f"Minimum Cost : {best_cost} km")


# =====================================================
# Main
# =====================================================

def main():

    routes = generate_routes()

    print_result(routes)


if __name__ == "__main__":
    main()