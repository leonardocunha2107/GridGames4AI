import numpy as np
import itertools
from status import *
        
class Checkers:
    EMPTY=0
    WHITE=1
    BLACK=2
    OTHER={WHITE:BLACK,BLACK:WHITE}
    def __init__(self):
        EMPTY_LINE=[self.EMPTY]*8
        line_1 =lambda color : [self.EMPTY,color]*4
        line_2=lambda color :  [color,self.EMPTY]*4
        self.__board=np.vstack([line_1(self.WHITE),line_2(self.WHITE),line_1(self.WHITE),EMPTY_LINE,
                                EMPTY_LINE,line_2(self.BLACK),line_1(self.BLACK),line_2(self.BLACK)])
        self.__turn=self.WHITE
        self.__is_queen=np.zeros((8,8))
        self.__pieces={self.WHITE:set([p for p in itertools.product(range(8),range(8)) if self.__board[p]==self.WHITE])}
        self.__pieces[self.BLACK]=set([p for p in itertools.product(range(8),range(8)) if self.__board[p]==self.BLACK])
        self.__can_backward=True
        self.turn_count=1
    def __change_turn(self):
        self.__turn=self.OTHER[self.__turn]
    def __queen_pos(self,pos0):
        cross=list(zip(range(1,8),range(1,8)))
        cross.extend(list(zip(range(1,8),range(-7,0))))
        cross.extend(list(zip(range(-7,0),range(1,8))))
        cross.extend(list(zip(range(-7,0),range(-7,0))))
        i0,j0=pos0
        return [(i0+ki,j0+kj) for ki,kj in cross if 0<=i0+ki<8 and 0<=j0+kj<8]
    def __norm_pos(self,pos0):
        i0,j0=pos0
        res=[]
        for i,j in [(i0+1,j0+1),(i0+1,j0-1),(i0-1,j0+1),(i0-1,j0-1)]:
            if 0<=i<8 and 0<=j<8:
              res.append((i,j))
        return res
    def __jumped_pos(self,pos0,pos1):
        def sign(n):
            if n>=0:
                return 1
            return -1
        i0,j0=pos0
        i1,j1=pos1
        si,sj=sign(i1-i0),sign(j1-j0)
        i,j=i0+si,j0+sj
        while True:
            if  self.__board[i,j]==self.OTHER[self.__turn]:
                break
            elif self.__board[i,j]==self.__turn:
                return False, None
            elif (i,j)==(i1,j1):
                return True,None
            i,j=i+si,j+sj
        if (i+si,j+sj)==(i1,j1):
            return True,(i,j)
        return False, None
        
    def __can_move_and_eaten(self,pos0,pos1):
        i0,j0=pos0
        i1,j1=pos1
        if self.__board[pos0]==self.__turn and ((0<=i1<=7) and (0<=j1<=7)) and self.__board[pos1]==self.EMPTY and abs(i1-i0)==abs(j1-j0) and pos0!=pos1:
            check,pos_eaten=self.__jumped_pos(pos0,pos1)
            if self.__is_queen[pos0] and check and not self.__can_backward:
                return True,pos_eaten
            elif self.__can_backward and pos_eaten and pos_eaten in self.__norm_pos(pos0):
                return True,pos_eaten
            elif self.__turn==self.WHITE and (pos1 in [(i0+1,j0-1),(i0+1,j0+1)] or (pos_eaten and pos_eaten in [(i0+1,j0-1),(i0+1,j0+1)])):
                return True,pos_eaten
            elif self.__turn==self.BLACK and (pos1 in [(i0-1,j0-1),(i0-1,j0+1)] or (pos_eaten and pos_eaten in [(i0-1,j0-1),(i0-1,j0+1)])):
                return True,pos_eaten
        return False,None
    def __posible_moves(self,pos0):
        res=[]
        i0,j0=pos0
        if self.__is_queen[pos0]:
            for pos in self.__queen_pos(pos0):
                check,pos_eaten=self.__can_move_and_eaten(pos0,pos)
                if check:
                    res.append((pos,pos_eaten))
        else:
            for ki,kj in zip([-2,-1,1,2],[-2,-1,1,2]):
                pos=(i0+ki,j0+kj)
                check,pos_eaten=self.__can_move_and_eaten(pos0,pos)
                if check:
                    res.append((pos,pos_eaten))
        return res
    def __can_eat(self,pos0):
        for pos,pos_eaten in self.__posible_moves(pos0):
            if pos_eaten:
                return True
        return False
    def __turn_queen(self,pos):
        i,j=pos
        if (self.__turn==self.WHITE and i==7) or  (self.__turn==self.BLACK and i==0):
            self.__is_queen[pos]=1
    def get_board(self):
        return self.__board
    def move(self,pos0,pos1):
        i0,j0=pos0
        if not (0<=i0<8 and 0<=j0<8):
            raise GameException("Invalid start position mate")
        if self.__board[pos0]!=self.__turn:
            raise GameException("Not your turn mate")
        eatable_pos=[p for p in self.__pieces[self.__turn] if self.__can_eat(p) ]
        
        check,eaten=self.__can_move_and_eaten(pos0,pos1)
        if not check:
            raise GameException("Invalid Move mate")
        if not eaten and eatable_pos:
            raise GameException("You need to jump another piece")
        
        player=self.__turn
        self.__board[pos0]=self.EMPTY
        self.__is_queen[pos0]=self.EMPTY
        self.__board[pos1]=player
        self.__pieces[player].remove(pos0)
        self.__pieces[player].add(pos1)
        
        if eaten:
            self.__pieces[self.OTHER[player]].remove(eaten)
            self.__board[eaten]=self.EMPTY
            self.__is_queen[eaten]=0
        if not self.__pieces[self.OTHER[player]]:
            s={self.WHITE:"Whites",self.BLACK: "Blacks"}
            raise GameFinish("{} won".format(s[player]))
        self.__can_backward=True
        if not self.__can_eat(pos1):
            self.__can_backward=False
            self.__turn=self.OTHER[player]
            self.turn_count+=1
    def ascii_board(self):
        symbol={self.EMPTY:'-',self.BLACK:'#',self.WHITE:'@'}
        res=["Turn: {}".format(self.turn_count),"Blacks: {}  Whites: {}".format(symbol[self.BLACK],symbol[self.WHITE])]
        res.append("It's {} turn \n".format(symbol[self.__turn]))
        line_idx=' '.join([str(i) for i in range(0,8)])
        res.append("  {}".format(line_idx))
        for i in range(8):
            res.append("{} {}".format(i,' '.join([symbol[t] for t in self.__board[i,:]])))
        return '\n'.join(res)
            
        
        