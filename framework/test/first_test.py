from examples.checkers.board import Checkers
from framework import AIGame
from framework.torchmodule import AITrainer

def test():
    trainer=AITrainer(AIGame(Checkers()))
    t=trainer.get_action()
    print(t)
    