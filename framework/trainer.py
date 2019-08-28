from .torchmodule import AIAgent
from .wrapper import AIGame
import torch
class AITrainer:
    def __init__(self,aigame:AIGame,**kwargs):
        self.game=aigame
        self.reg=kwargs.get('reg',0)
        lr=kwargs.get( 'lr',1e-3)
        
        self.agent=AIAgent(aigame.n_pieces,aigame.grid_shape,aigame.action_range)
        self.optim=kwargs.get('optim',torch.optim.Adam)(self.agent.parameters(),lr=lr)#,weight_decay=self.reg)
        self.__high=torch.tensor([a[1] for a in aigame.action_range],dtype=torch.float)
        self.__low=torch.tensor([a[0] for a in aigame.action_range],dtype=torch.float)
        self.__move_loss=torch.nn.L2Loss()
    def __try_action(self):
        def __decode(action):
            x= (torch.min(action,self.__high)+torch.max(action,self.__low))/2
            return  tuple(x.type(torch.int).tolist())
        state=self.game.get_board()
        action= self.agent(state)
        move=__decode(action)
        print(move)
        return action,self.game.try_move(move)
    
    def one_step(self):
        move_count=0
        while True:
            move_count+=1
            action,valid=self.__try_action()
            if not valid:
                valid_action=self.game.get_actions()[0]
                loss=self.__move_loss(action,valid_action)
                loss.backward()
                self.optim.step()
            else:
                return move_count,action

        