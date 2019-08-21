from board import Checkers

g=Checkers()
def move(pos0,pos1):
    g.move(pos0,pos1)
    print(g.ascii_board())
move((2,1),(3,2))
move((5,0),(4,1))
move((3,2),(5,0))
#move((5,2),(4,1))
#move((2,3),(3,2))
#move((5,4),(4,3))