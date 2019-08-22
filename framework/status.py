class GameException(Exception):
    def __init__(self,message):
        super(GameException,self).__init__(message)
class GameFinish(Exception):
    """
    Indicates to Wrapper that we've arrived at an end state
    
    Parameters:
    -------------
        winner_str: str
            Name of the winner
        winner: int
            Number of the winner
            Defaults to None, which represents a tie
        ponct: 
            Ponctuation of the game if it exists
    """
    def __init__(self,winner_str=None,winner=None,ponct=None):
        if not winner:
            msg="The game tied"
        elif not winner_str:
            msg="Someone won"
        else:
            msg="{} won".format(winner_str)
        super(GameFinish,self).__init__(msg)
        

