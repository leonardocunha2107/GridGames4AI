"""
    Does the encoding of the game state and actions based on seqtoseq approach
"""
##TODO: Implement RL compatible
from .wrapper import AIGame
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CategoricalConv(nn.Module):
   """
       Traditional Conv2D, but it has a set of filters for each 'piece'
       By it's nature,it's not differentiable on the inputs, so it has the implicated limitations
       Parameters
       -------------
           n_pieces: int
               Number of different elements of the input, in a grid game that is the number of pieces. 
               Any input x has to have values in range(n_pieces)
           grid_shape: tuple(int,int)
               Shape of inputs
           filters_by_piece: int
               Number of filters by piece, that is, outpus will have n_pieces*filters_by_pieces channels.
           padding: int
               Same as for Conv2D
           stride: int
               Same as for Conv2D
           kernel_size: tuple(int,int)
               Same as for Conv2D

   
   """
   def __init__(self,n_pieces,grid_shape,padding=0,kernel_size=(3,3),filters_by_piece=2,stride=1):
       super(CategoricalConv,self).__init__()
       self.convs=[nn.Conv2d(in_channels=1,kernel_size=kernel_size,out_channels=filters_by_piece,padding=padding) for i in range(n_pieces)]
       self.grid_shape=grid_shape
       self.__ones=torch.tensor(np.ones(grid_shape,dtype=int))
       self.__zeros=torch.tensor(np.zeros(grid_shape,dtype=int))
     
   def forward(self,x):
       """
           Parameters:
               x: torch.tensor(grid_shape)
                   Input grid
           Returns: torch.tensor(n_pieces,filters_by_piece,*grid_shape)
               
       """
       aux=[conv(torch.where(x==k,self.__ones,self.__zeros).view(1,1,*self.grid_shape).type(torch.FloatTensor)) for k,conv in enumerate(self.convs)]
       return torch.cat([t.view(1,*t.shape[1:]) for t in  aux],dim=0)

class GridEncoder(nn.Module):
    """
       Encoder for grid
       Parameters
       -------------
           n_pieces: int
               Number of different elements of the input, in a grid game that is the number of pieces. 
               Any input x has to have values in range(n_pieces)
           out_dim: int
               dimension of output encoding
           grid_shape: tuple(int,int)
               Shape of inputs
           filters_by_piece: int
               Number of filters by piece, that is, outpus will have n_pieces*filters_by_pieces channels. Defaults to 2
           padding: int
               Same as for Conv2D. Defaults to 0
           stride: int
               Same as for Conv2D. Defaults to 1
           kernel_size: tuple[int,int]
               Same as for Conv2D. Defaults to (3,3)
           intermediate_convs: int
               Number of intemediate convolutions. Defaults to 2"""
    def __init__(self,n_pieces,grid_shape,out_dim,**kwargs):
       
               

   
          
        super(GridEncoder,self).__init__()
        
        self.kernel_size=kwargs.get('kernel_size',(3,3))
        self.padding=kwargs.get('padding',0)
        self.stride=kwargs.get( 'stride',1)
        self.filters_by_piece=kwargs.get('filters_by_piece',2)
        self.n_pieces=n_pieces
        self.grid_shape=grid_shape
        n_convs=kwargs.get('intermediate_convs',2)
        
        self.categorical_conv=CategoricalConv(n_pieces,grid_shape,padding=self.padding,
                                              kernel_size=self.kernel_size,filters_by_piece=self.filters_by_piece,stride=self.stride)
        channels=self.n_pieces*self.filters_by_piece
        self.other_convs=nn.Sequential(*tuple([nn.Conv2d(channels,channels,self.kernel_size,padding=self.padding,stride=self.stride),
                                        nn.ReLU()]*n_convs))
        
        h,w=grid_shape
        for _ in range(n_convs+1):
            h=int((h+2*self.padding-self.kernel_size[0])/self.stride+1)
            w=int((w+2*self.padding-self.kernel_size[1])/self.stride+1)
     
        self.linear=nn.Linear(h*w*channels,out_dim)

    def forward(self,x):
        x=self.categorical_conv(x)
        x=F.relu(x.view(1,self.n_pieces*self.filters_by_piece,*x.shape[2:]))
        x=self.other_convs(x).view(1,-1)
        return self.linear(x)
        
        
class AIActuator(nn.Module):
    def __init__(self):
        super(AIActuator,self).__init__()
        self.v=nn.Linear(10,10)
    def forward(self,x):
        a=torch.tensor([1,2])
        a.vi
        return self.v(x)