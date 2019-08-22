class GameException(Exception):
    def __init__(self,message):
        super(GameException,self).__init__(message)
class GameFinish(Exception):
    def __init__(self,winner_str,winner,ponct='None'):
        super(GameFinish,self).__init__(message)
        

