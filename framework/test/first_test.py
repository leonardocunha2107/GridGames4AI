from examples.checkers.board import Checkers
from framework import AIGame
from framework.torchmodule import AIAgent
from framework.trainer import AITrainer
import numpy as np
import torch
from time import time

def test1():
    ag=AIAgent(4,(8,8),[(10,10)])
    a=np.arange(4).reshape(2,2)
    a=np.hstack((a,a,a,a))
    board=torch.tensor(np.vstack((a,a,a,a)))
    print(ag(board))
def test2():
    tr=AITrainer(AIGame(Checkers()))
    move_action=[]
    start=time()
    for i in range(100):
        ma=tr.one_step()
        print(f'Did {i} move')
        move_action.append(ma)
    print('Total time {}'.format(time()-start))
    return move_action