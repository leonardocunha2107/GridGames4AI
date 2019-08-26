"""
    Does the encoding of the game state and actions based on seqtoseq approach
"""
##TODO: Implement RL compatible
from .wrapper import AIGame
import torch.nn as nn
import torch

class CategoricalConv(nn.Module):
   """
   traditional Conv2D, but it has a set of filters for each 'piece'
   By it's nature,it's not differentiable on the inputs, so it has the implicated limitations
   """
   def __init__(self,n_pieces,padding=0,kernel_size=3,filters_by_piece=2,stride=0):
       self.convs=[nn.Conv2d(in_channels=1,kernel_size=kernel_size,out_channels=filters_by_piece,padding=padding) for i in range(n_pieces)]

   def forward(self,x):
       aux=[conv(torch.where(x==k,1,0)) for k,conv in enumerate(self.convs)]
       return torch.cat([t.view(1,*t.shape) for t in  aux],dim=0)

class StateEncoder(nn.Module):
    def __init__(self,grid_shape,num_pieces):
        pass
        
class AIActuator(nn.Module):
    def __init__(self):
        self.v=nn.Linear(10,10)
    def forward(self,x):
        return self.v(x)