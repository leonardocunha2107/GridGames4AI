from board import Checkers
import numpy as np
board=np.zeros((8,8))
board[7,0]=1
board[5,2]=2
is_queen=[(7,0)]
g=Checkers()
g.set_board(board,is_queen=is_queen)
print(g.ascii_board())