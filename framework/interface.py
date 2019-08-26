##Let's use ducktype instead of 
class InterfaceException(Exception):
    pass
class GridInterface:
    """
    A minimal interface a game has to implement to communicate with AI
    
    Methods
    -----------
    Required:
        __init__:
            Sets the state to the initial state. 
            Must have a set of default values to assert proper implementation
        get_board: 
            Returns:  numpy.array(int)
            Must return a 2D representatoin of the game state
        get_spec: 
            Returns: dict
            Must dictionary with game specifications
        get_actions:
            Returns: list(tuple(int))
            Must return a list of possible actions in current game state, each action represented by a tuple
        
        get_turn: 
            Returns: int
            Return int representation of wh's turn it is in the current state
            
        move(tuple(int)):
            Execute action represented by tuple, raising GameException if not possible or GameFinish if arrived at end state
        
    Optional:
        __str__:
            Returns: str
            Get visualization of the current state
        get_reward:
            Returns: float or np.array(float)
            Get ponctuation based on game ste or history, for one or each player
        
    """
    def __init__(self):
        """
            Sets the state to the initial state. 
            Must have a set of default values to assert proper implementation
        """
        raise InterfaceException("__init__ not implemented")
    def get_board(self):
        """
        Returns:  numpy.array(int)
        Must return a 2D representatoin of the game state
        """
        raise InterfaceException("get_board not implemented")
    
    def get_spec(self):
        """
        Returns: dict
        Get game specifications. Can be loaded from a json for example4
        Required parameters:
            'players': dict[int,list[int]]
                A dict tying each player code, the same we would get in get_turn
                and a list of pieces that belong to him
                If we have a symmetric game, the pieces codes shold be ordered in the same way
            'env_spaces': list[int]
                List of tokens that represent an environment space, e.g. wall or empty space
            'action_range': tuple(tuple(int))
                Tuple of 'ranges' indicating the range of values an action can take
                E.g, for  chess or checkers it would be ((0,8),(0,8),(0,8),(0,8))
            
        Optional parameters:
            'symmetric': bool
                Indicates if a game is symmetric, e.g. chess, that is if each player has pieces with same signification. Defaults to true
            
        """
        raise InterfaceException("get_conf not implemented")
    
    def get_actions(self):
        """
            Returns: list(tuple(int))
            Must return a list of possible actions in current game state, each action represented by a tuple

        """
        raise InterfaceException("get_actions not implemented")
        
    def get_turn(self):
        """
            Returns: int
            Return int representation of wh's turn it is in the current state

        """
        raise InterfaceException("get_turn not implemented")
    def move(self,tup):
        """
             Execute action represented by tuple, raising GameException if not possible or GameFinish if arrived at end state

        """
        raise InterfaceException("move(tuple) not implemented")
    def __str__(self):
        """
            Returns: str
            Get visualization of the current state
        """
        return "\n\nNO BOARD VISUALIZATION MADE0\n\n"
    def get_reward(self):
        """
            Returns: float or np.array(float)
            Get ponctuation based on game ste or history, for one or each player

        """
        raise InterfaceException("get_reward not implemented")
    