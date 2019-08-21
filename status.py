class GameException(Exception):
    def __init__(self,message):
        super(GameException,self).__init__(message)
class GameFinish(Exception):
    def __init__(self,message):
        super(GameFinish,self).__init__(message)
        

