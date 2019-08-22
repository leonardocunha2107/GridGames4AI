##Let's use ducktype instead of 
class InterfaceException(Exception):
    pass
class GridInterface:
    """
    A minimal interface a game has to implement to communicate with AI
    
    Methods
    -----------
    Required:
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
    def get_board(self):
        """
        Returns:  numpy.array(int)
        Must return a 2D representatoin of the game state
        """
        raise InterfaceException("get_board not implemented")
    
    def get_spec(self):
        """
        Returns: dict
        Required parameters:
            TODO
        Optional paramaeters:
            TODO
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
    