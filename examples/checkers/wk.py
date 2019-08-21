from board import Checkers

g=Checkers()
g.move((2,1),(3,2))
g.move((5,0),(4,1))
g.move((3,2),(5,0))
g.move((5,2),(4,1))
g.move((2,3),(3,2))
print(g.ascii_board())
g.move((5,4),(4,3))