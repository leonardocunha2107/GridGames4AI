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
       self.__ones=torch.ones(grid_shape,dtype=torch.int)
       self.__zeros=torch.zeros(grid_shape,dtype=torch.int)
     
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
        
class MultilayerLSTMCell(nn.Module):
    def __init__(self,input_dim,n_layers):
        super(MultilayerLSTMCell,self).__init__()
        self.input_dim=input_dim
        self.layers=[nn.LSTMCell(input_dim,input_dim) for _ in range(n_layers)]
        
        
        """
            Parameters
            ---------
                t: tuple(tensor,list[tuple(tensor)])
                    The input t corresponds to a tuple list of hidden and cell states for each layer
            
            Returns: tuple(list[tensor],list[tensor])
                    Same as input
        """
    def forward(self,t):
        
         x,hc_in=t
         assert (len(hc_in)==len(self.layers) and x.shape==(1,self.input_dim))
         hc_out=[]
         hc=None
         for i in range(len(self.layers)):
             x,hc=self.layers[i](x,hc_in[i])
             hc_out.append(hc)
         return hc_out

class ActionDecoder(nn.Module):
    ##TODO: eliminate action_range
    def __init__(self,input_dim,action_range):
        super(ActionDecoder,self).__init__()
        self.l1=nn.Linear(input_dim,len(action_range))
        """self.l2=nn.Linear(int(input_dim/2),len(action_range))
        self.norm=nn.LayerNorm(len(action_range))
        self.high=torch.tensor([a[1] for a in action_range],dtype=torch.float)
        self.low=torch.tensor([a[0] for a in action_range],dtype=torch.float)"""
    def forward(self,x):
        ##x=self.norm(self.l1(x))
        #x=(torch.min(x,self.high)+torch.max(x,self.low))/2
        return self.l1(x)

class AIAgent(nn.Module):
    def __init__(self,n_pieces,grid_shape,action_range,**kwargs):
        super(AIAgent,self).__init__()
        self.num_lstm_layers=kwargs.get('lstm_layers',3)
        self.embedding_dim=kwargs.get('embedding_dim',90)
        self.num_decoder_layers=kwargs.get('decoder_layers',3)
        
        self.encoder=GridEncoder(n_pieces,grid_shape,self.embedding_dim)
        self.agent=MultilayerLSTMCell(self.embedding_dim,self.num_lstm_layers)
        self.decoder=ActionDecoder(self.embedding_dim,action_range)
        self.hc=[(torch.zeros((1,self.embedding_dim)),torch.zeros((1,self.embedding_dim))) \
                 for _ in range(self.num_lstm_layers)]
    def forward(self,x):
        embed=self.encoder(x)
        hc=self.agent((embed,self.hc))
        action=self.decoder(hc[-1][0])
        return action
        

        