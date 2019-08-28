import numpy as np
import torch
from .interface import GridInterface,InterfaceException
from .status import GameException
class AIGame:
    """
    Enforces GridInterface proper implementaton and wraps the game into an AI
    playable  series of sessions
    
    """
    def __init__(self,game: GridInterface,max_turns=100):
        """
        Parameters:
            game: GridInterface
                Instance of a class that implements GridInterface
            max_turns: int
                Maximum number of rounds till we get a tie
                
        """
        ##DuckTyping
        try:
            for f in [game.get_board,game.get_spec,game.get_turn,game.get_actions,game.__init__]:
                _=f()
            #game.move(None)
        except InterfaceException:
            raise TypeError("The game's interface is not properly implemented")
        
        spec=game.get_spec()
        try:
            self.pieces_toks=spec["players"]
            self.players=self.pieces_toks.keys()
            self.env_toks=spec['env_spaces']
            self.action_range=spec['action_range']
        except KeyError:
            raise TypeError("The function get_spec doesn't give sufficient specification")
        
        self.grid_shape=game.get_board().shape
        self.__max_turns=max_turns
        
        self.symmetric=spec.get("symmetric",True)
        if self.symmetric:
            a=[len(pieces) for pieces in self.pieces_toks.values()]
            if max(a)!=min(a):
                raise TypeError("Game is set to symmetric but players don't have the same number of pieces")
                
        ##---end of  validation
        self.game=game
        self.board=None
        self.__flipped=None
        
        ##TODO: Do it properly
        self.piece_dict=None
        if self.symmetric:
            self.piece_dict={0:0,1:3,2:4,3:1,4:2}
        
        self.n_pieces=sum([len(pieces) for pieces in self.pieces_toks.values()])
        self.__reset()
    def __reset(self):
        self.game.__init__()
        self.turn=0
        self.board=self.game.get_board()
        self.__flipped=False
    def get_actions(self):
        return [torch.tensor(move,dtype=torch.float) for move in self.game.get_actions()]
    
    def __flip(self):
        ##TODO: Flip properly
        if len(self.players)!=2 or not self.symmetric:
            return
    
        board=self.game.get_board()
        if not self.__flipped:
            m,n=board.shape
            for i in range(int(m/2)):
                for j in range(n):
                    board[i,j],board[m-i-1,n-j-1]=self.piece_dict[board[m-i-1,n-j-1]],self.piece_dict[board[i,j]]
        self.__flipped=not self.__flipped
        self.board=board
                
    def get_board(self):
        return torch.tensor(self.board)
    def try_move(self,move):
        assert len(move)==len(self.action_range)
        if self.__flipped:
            #TODO: do it correctly
            move=tuple(7-move[i] for  i in range(len(move)))
        try:
            self.game.move(move)
            self.__flip()
            return True
        except GameException:
            return False
    
        
        
            