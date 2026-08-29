### [Page 1]
데이터구조
박경훈
11주차: AVL트리코드구현



### [Page 2]
※ 아래데이터를기준으로코드설명
- 데이터: 43, 49, 84, 12, 63, 69, 96, 89 
43
STEP 1
DATA
left
right
height
- Node 구조
- left: none
- right: none
- height: 1
43
STEP 2
- left: none
- right: “49” node
- height: 2
49
- left: none
- right: none
- height: 1
get_balance() = -1
왼쪽노드높이값– 오른쪽노드높이값


| DATA |  |  |
| --- | --- | --- |
| left | right | height |


### [Page 3]
- 데이터: 43, 49, 84, 12, 63, 69, 96, 89 
43
STEP 3
- left: none
- right: “49” node
- height: 3
49
- left: none
- right: “84” node
- height: 2
get_balance() = -2
84
- left: none
- right: none
- height: 1
get_balance() = -1
- 불균형발생
- 음수값이고, 49 < 84 이므로
- RR 회전동작



### [Page 4]
- 데이터: 43, 49, 84, 12, 63, 69, 96, 89 
Continue
STEP 3
rotate_left(x)
49
none
“84” 노드
2
43
none
“49” 노드
3
y
x
T2
49
x
“84” 노드
2
y
43
none
T2
1
x
84
none
none
1
43
49
84


| 49 |  |  |
| --- | --- | --- |
| none | “84” 노드 | 2 |

| 49 |  |  |
| --- | --- | --- |
| x | “84” 노드 | 2 |

| 43 |  |  |
| --- | --- | --- |
| none | T2 | 1 |

| 84 |  |  |
| --- | --- | --- |
| none | none | 1 |


### [Page 5]
- rotate_left 를참조하여rotate_right를임의데이터를이용하여도식화해보세요. (코드참조) 
rotate_right (x)

