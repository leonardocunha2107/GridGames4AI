"""
Simple cli_game based on the CheckersBase class  
Input can be either '(i0,j0,i1,j1)' or (i0,j0),(i1,j1) 
(i0,j0) being the initial position of the piece to be moved and (i1,j1) the end position
Enter 'exit' to quit
"""
from .board import Checkers
import sys
import time
from framework import *
from os import system,name

def clear(): 
   ##Thanks stackoverflow
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
        
if __name__=='__main__':
    game=Checkers()
    while (True):
        print("{}\n\n".format(game))
        move=input("Move:\n")
        if move=='exit':
            sys.exit(0)
        try:
            aux=eval(move)
            try:
                if type(aux[0])==tuple:
                    game.move(a[0]+a[1])
                else:
                    game.move(a)
            except GameException as e:
                clear()
                print(e)
                time.sleep(1)
            except GameFinish as e:
                clear()
                print(e)
                time.sleep(5)
                sys.exit()
        except Exception as e:
            print("Incorrect syntax")