import random
import math
import time
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

CELL = 40 #Number of pixels per display cell
WIDTH = 5 #Number of cells wide screen
############################## Tkinter Initialization ##############################
window = Tk()
window.resizable(False, False)
canvas = Canvas(window,
                bg = "black",
                width = WIDTH*CELL,
                height = WIDTH*CELL + 280)
canvas.pack()
############################################################
############################## Block Class ##############################
############################################################
class Block:
    ############################## Block Constructor ##############################
    def __init__(self, x, y, c, o):
        self.loc = (x, y) #x, y coordinates
        self.obj = canvas.create_rectangle(x*CELL, 
                                           y*CELL, 
                                           (x+1)*CELL, 
                                           (y+1)*CELL, 
                                           fill = c, 
                                           outline = o) #Tkinter canvas widget object
    ############################## Block Constructor ##############################
    def __del__(self):
        canvas.delete(self.obj)
############################################################
############################## Snake Class ##############################
############################################################
class Snake:
    ############################## Snake Constructor ##############################
    def __init__(self):
        self.dir = random.randint(0, 3) #Direction
        self.blocks = [Block(random.randint(0, WIDTH-1), 
                             random.randint(0, WIDTH-1), 
                             "lime", 
                             "black")] #Body
        
    ############################## Move ##############################
    #Add a block in location next to head in "direction". Delete 
    #tail block if apple not eaten in last move.
    def move(self, ate):
        head = self.blocks[0].loc
        #Add new block next pos in direction of head
        if self.dir == 0:
            self.blocks.insert(0, Block(head[0]+1, head[1], "lime", "black"))
        elif self.dir == 1:
            self.blocks.insert(0, Block(head[0], head[1]+1, "lime", "black"))
        elif self.dir == 2:
            self.blocks.insert(0, Block(head[0]-1, head[1], "lime", "black"))
        elif self.dir == 3:
            self.blocks.insert(0, Block(head[0], head[1]-1, "lime", "black"))
        if not ate: #If ate, keep extra block
            #Else remove block each move
            if len(self.blocks):
                pop = self.blocks.pop()
                del pop
    ############################## Set Dir ##############################
    #Change direction if passed opposite parity
    def setDir(self, new):
        if (self.dir % 2) != (new % 2): #Direction can only change to opposite parity
            self.dir = new % 4
    ############################## On Snake ##############################
    #Returns True if passed block shares location with a body block
    def onSnake(self, blockIn):
        if len(self.blocks):
            for block in self.blocks[1:]:
                if blockIn.loc == block.loc:
                    return True
        return False
############################## Check Ate ##############################
#Returns True if head is on apple
def checkAte(snake, apple):
    if len(snake.blocks):
        return snake.blocks[0].loc == apple.loc
############################## Check OB ##############################
#Returns True if head is out of bounds
def checkOB(snake):
    if len(snake.blocks):
        return (snake.blocks[0].loc[0] < 0 or
        snake.blocks[0].loc[0] > WIDTH or
        snake.blocks[0].loc[1] < 0 or
        snake.blocks[0].loc[1] > WIDTH)
def checkWin(snake):
    return len(snake.blocks) == WIDTH * WIDTH
############################## Check Self-Collision ##############################
#Returns True if head is on body
def checkSelfCollision(snake):
    return snake.onSnake(snake.blocks[0]) #Returns true of snake's head is on a body block
############################################################
############################## Main ##############################
############################################################
if __name__ == "__main__":
    #Create snake
    snake = Snake() 
    #Bind arrow keys
    window.bind("<Right>", lambda event: snake.setDir(0))
    window.bind("<Down>", lambda event: snake.setDir(1))
    window.bind("<Left>", lambda event: snake.setDir(2))
    window.bind("<Up>", lambda event: snake.setDir(3))
    #Create apple
    apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black")
    #Animation loop
    while True:
        time.sleep(.1)
        ate = checkAte(snake, apple) #Check if head on apple
        if ate:
            del apple 
            apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black") #Move apple
        snake.move(ate) #Move snake
        window.update()