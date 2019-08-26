import numpy as np
from .interface import GridInterface,InterfaceException
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
            game.move(None)
        except InterfaceException:
            raise TypeError("The game's interface is not properly implemented")
        
        spec=game.get_spec()
        try:
            self.pieces_toks=spec["players"]
            self.players=self.pieces.keys()
            self.env_toks=spec['env_spaces']
            self.action_range=spec['action_range']
        except KeyError:
            raise TypeError("The function get_spec doesn't give sufficient specification")
        
        self.grid_shape=game.get_board().shape
        self.max_turns=max_turns
        
        self.symmetric=spec.get("symmetric",True)
        if self.symmetric:
            a=[len(pieces) for pieces in self.pieces_toks]
            if max(a)!=min(a):
                raise TypeError("Game is set to symmetric but players don't have the same number of pieces")
        
        
            