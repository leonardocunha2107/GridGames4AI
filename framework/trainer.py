from .torchmodule import AIAgent
from .wrapper import AIGame
    
class AITrainer:
    def __init__(self,aigame:AIGame,**kwargs):
        self.game=aigame
        self.agent=AIAgent(aigame.n_pieces,aigame.grid_shape,aigame.action_range)
    def try_action(self):
        state=self.aigame.get_board()
        return self.agent(state)