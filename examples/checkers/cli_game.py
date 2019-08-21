from board import Checkers
import sys
import time
from status import *
from os import system,name

def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
if __name__=='__main__':
    game=Checkers()
    while (True):
        print("{}\n\n".format(game.ascii_board()))
        move=input("Move:\n")
        try:
            p0,p1=eval(move)
            try:
                game.move(p0,p1)
            except GameException as e:
                clear()
                print(e)
                time.sleep(1)
            except GameFinish as e:
                clear()
                print(e)
                time.sleep(5)
                sys.exit()
            except KeyboardInterrupt:
                sys.exit()
        except Exception as e:
            print(e.with_traceback())