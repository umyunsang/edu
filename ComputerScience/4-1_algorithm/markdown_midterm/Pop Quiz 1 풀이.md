# Pop Quiz 1 풀이

- Source PDF: `ComputerScience/4-1_algorithm/PopQuiz/Pop Quiz 1 풀이.pdf`
- Total pages: 4

## Page 1

Pop Quiz #1
[ 1]

Question

, ?( ,
.)

def hanoi_tower(n, fr, tmp, to) :
if (n == 1) :
print(" 1: %s --> %s" % (fr, to))
else :
hanoi_tower(n - 1, fr, to, tmp)
print(" %d: %s --> %s" % (n, fr, to))
hanoi_tower(n - 1, tmp, fr, to)

(PDF )

:

(p.33) :

:
:

:

(1) , (2) , (3)
.
.

[ 2]

Question

,
?

(PDF )

:

## Page 2

: .

( )

1.

, ( ) .
( : )

2. (Recursion Tree)

Level 0: ( : )
Level 1: ( : )
Level 2: ( : )
2 .
( ) , .

3. (Master Theorem)

.
, .
Case 1 , .

[ 3]

Question

0 1 text 20 , 19 0, 1 .
0000...000001
5 00001 , ?

(PDF )

5 .
text: 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 1
pattern: 0 0 0 0 1
, 5 .
.
text: 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 1
pattern: 0 0 0 0 1
, 15 , 16 X 5 .

(Brute-force) . .
( 00001 ) , 5 .
.

[ 4]

## Page 3

Question

, ?

(PDF )

:
. .

[ 5]

Question

, ?

(PDF )

:
. .

(Complexity Hierarchy)

( , )

( ): . .
( ): .
.

(Big-O): (Upper Bound). ( ). "
."
(Big-Omega): (Lower Bound). ( ). "
."

## Page 4

(Big-Theta): (Tight Bound). ( ). "
."

True/False

→ True

( ) . ( )
.

→ True

( ) . 1 ( ) 2 ( ) .
.

→ False

( ) ( ) .
.

→ False

. ( ) ( )
.

→ False

( ) . .
.

tree . → True

' ' . .

tree . → True

,
.

graph tree . → False

. (Connected) .
(Forest) .

## Page 5

(텍스트 없음)
