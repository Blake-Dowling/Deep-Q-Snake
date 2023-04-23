import random
import math
import time
from tkinter import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import snake_game
from snake_game import *
import visual.visual
from visual.visual import *
import visual.embed_plot
import ast




#Assigns object and distance value for each direction and highlights blocks seen
def look(snake, apple):
    global blocksSeen #Block objects seen (white outlined visual)
    #Reset snake's seen blocks
    if len(blocksSeen):
        for block in blocksSeen:
            del block
    if len(snake.blocks): #Check snake has at least one block
        head = snake.blocks[0].loc #Snake head coordinates
    else:
        return np.array([0.0 for i in range(16)])
    distToWall = [WIDTH-head[0],
                   WIDTH-head[1],
                   head[0]+1,
                   head[1]+1
] #Distance to wall in each direction 
    seen = [[] for i in range(4)] #Contains a list of seen blocks for each direction
    object = [0.0 for i in range(4)] #Object seen in each direction
    wall = [0.0 for i in range(4)] #Wall seen in each direction
    body = [0.0 for i in range(4)]
    for direction in range(0, 4): #For each direction
        for radius in range(1, distToWall[direction]): #For each coordinate from head to wall in current direction,
            # where radius is radial distance from head
            match (direction):
                case 0: nextLoc = (head[0]+radius, head[1]) #Block coordinates seen to right
                case 1: nextLoc = (head[0], head[1]+radius) #Block coordinates seen to bottom
                case 2: nextLoc = (head[0]-radius, head[1]) #Block coordinates seen to left
                case 3: nextLoc = (head[0], head[1]-radius) #Block coordinates seen to top
                case _: break
            seen[direction].append(nextLoc) #Blocks seen so far in direction
            ##########Check if body block next to head##########
            # if radius == 1: #Vision block coordinate next to head in currently checked direction
            #     for block in snake.blocks: #Must check each body block
            #         if block.loc == nextLoc: #Seen block is a snake body block
            #             body[direction] = 1.0 #Set input for 'body seen in this direction' to 1.0
            #             break
        ##########Check if wall is next to head in currently checked direction##########
        if len(seen[direction]) == 0 and (direction % 2) == 0: #If vision length is 0 in direction and direction is
            #right, down, left, or up
            wall[int(direction / 2)] = 1.0 #Set wall to 1 for that direction
    ##########Check apple's location in relation to snake head##########
    if head[0] < apple.loc[0]: #Apple to right of snake head
        object[0] = 1.0
    if head[1] < apple.loc[1]: #Apple below snake head
        object[1] = 1.0
    if head[0] > apple.loc[0]: #Apple to left of snake head
        object[2] = 1.0
    if head[1] > apple.loc[1]: #Apple above snake head
        object[3] = 1.0
    ##########Handle drawing white block outlines##########
    blocksSeen = [] #Reset blocksSeen list
    for seenDirection in seen:
        for seenBlock in seenDirection:
            blocksSeen.append(Block(seenBlock[0], seenBlock[1], "", "white"))
    return tf.constant([object + wall])# + distance])

def updatePlot(statsPlot, statsCanvas, train_stats_x, train_stats_y, iteration, apples, fails):
    ##########Update stats plot x data##########
    cumulative_iterations = iteration #cumulative iterations (x-axis)
    if len(train_stats_x):
        cumulative_iterations = train_stats_x[len(train_stats_x) - 1] + iteration #Add iterations from this
        #run to previous cumulative amount
    train_stats_x.append(cumulative_iterations) #Add cumulative iterations amount to plot's x data

    ##########Update stats plot y data##########
    performance = apples
    if fails > 0:
        performance = apples/fails
    train_stats_y.append(performance) #Add performancy to plot's y data

    ##########Update plot##########
    newLine, = statsPlot.plot(train_stats_x, train_stats_y)
    statsCanvas.draw()
    snake_game.canvas.update()
    newLine.remove()
    ##########Write Stats to scores file##########
    scoreFile = open("Scores1.txt", "a+")
    scoreFile.write("Train_Stats:" + str(train_stats_x) + ":" +\
                    str(train_stats_y) + ": \n" +\
                    "Cumulative_Iterations: " + str(iteration) +\
                    " Apples: " + str(apples) +\
                    " Fails: " + str(fails) +\
                    " Performance_(Apples/Fails): " + str(performance)
                    + " \n")
    scoreFile.close()

def loadStats():
    scoreFile = open("Scores1.txt", "r")
    lines = scoreFile.readlines()
    secondLastLine = ""
    lastLine = ""
    if(len(lines)):
        secondLastLine = lines[len(lines)-2]
        lastLine = lines[len(lines)-1]


    trainList = secondLastLine.split(':')
    lastLine.replace(':', ' ')
    lineList = lastLine.split(' ')
    train_stats_x = []
    train_stats_y = []
    if(len(trainList) >= 3):
        train_stats_x = list(map(float, ast.literal_eval(trainList[1])))
        train_stats_y = list(map(float, ast.literal_eval(trainList[2])))
        
    iteration = 0
    apples = 0
    fails = 0
    if(len(lineList) >= 9):
        iteration = int(lineList[1])
        apples = int(lineList[3])
        fails = int(lineList[5])
    scoreFile.close()
    return (train_stats_x, train_stats_y, iteration, apples, fails)
    

if __name__ == "__main__":
    

    snake = Snake()
    #Create apple
    apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black")
    
    blocksSeen = []

    input_train = []
    output_train = []

    #Bind arrow keys
    window.bind("<Right>", lambda event: snake.setDir(0))
    window.bind("<Down>", lambda event: snake.setDir(1))
    window.bind("<Left>", lambda event: snake.setDir(2))
    window.bind("<Up>", lambda event: snake.setDir(3))
    ####################New Model####################
    # model = keras.Sequential()
    # layer0 = keras.layers.Flatten(input_shape=([8]))
    # model.add(layer0)
    # layer1 = keras.layers.Dense(16, activation="relu")
    # model.add(layer1)
    # layer2 = keras.layers.Dense(4, activation="softmax")
    # model.add(layer2)
    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])




    ####################Load Model####################
    model = keras.models.load_model("model1.h5")

    ####################Tensor Visualizers####################
    # visualPlot1 = visual.visual.MatrixPlot(window, 0, 200)
    # visualPlot2 = visual.visual.MatrixPlot(window, 0, 450)

    ####################Main Loop####################
    TRAIN_MINUTES = 10
    iteration = 0
    loopedMoves = 0
    apples = 0
    fails = 0

    train_stats_x = []
    train_stats_y = []
    #Set stats to cumulative stats from Scores1 file
    (train_stats_x, train_stats_y, iteration, apples, fails) = loadStats()
    
    statsPlot, statsCanvas = embed_plot.embedPlot(window, 0, 0, 2, train_stats_x, train_stats_y)
    while iteration < 1000*TRAIN_MINUTES:
        if iteration % 100 == 0:
            updatePlot(statsPlot, statsCanvas, train_stats_x, train_stats_y, iteration, apples, fails)
        iteration = iteration + 1
        loopedMoves = loopedMoves + 1
        
        time.sleep(.005)
        #print("Iteration: " + str(iteration))
        
        ate = checkAte(snake, apple) #Check if head on apple
        if ate:
            apples = apples + 1
            loopedMoves = 0
            del apple 
            apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black") #Move apple
            for i in range(max(0,   min(len(input_train) -16, len(output_train) -16)), min(len(input_train), len(output_train))): #For each training frame
                dir = tf.get_static_value(output_train[i]) #(Good) direction of snake with current frame
                model.fit(input_train[i], tf.constant([dir]), epochs = 1, verbose=0) #Train each training state with direction frame
            input_train = []
            output_train = []
        
        #Append a tensorflow input based on snake's seen objects and their distances for each of eight directions
        currentVision = look(snake, apple) #Create current training frame
        input_train.append(currentVision) #Save each vision input for training (If successful)
        
        tfOutput = model.predict(currentVision, verbose=0) #Get tf model's best direction guess with current state
        tfOutput[0][(snake.dir + 2) % 4] = 0.0 #Set prediction in direction opposite to snake's direction to 0
        #to prevent 180 degree turns
        # visualPlot1.plotMatrix(currentVision) 
        # visualPlot2.plotMatrix(tfOutput)
        # print(tfOutput)
        
        bestDirection = tf.get_static_value(tf.math.argmax(tfOutput[0], output_type=tf.int64)) #Get max value of tf vector
        snake.dir = bestDirection #Set snake's new direction
        #Random direction every 16 frames to prevent looped learning
        if iteration % 16 == 0:
            snake.dir = random.randint(0,3)


        
        directionTensor = tf.constant(float(bestDirection)) #Training direction for current state
        output_train.append(directionTensor) #Save each chosen direction output for training (If successful)
        window.update()
        snake.move(ate) #Move snake
        snakeIsOB = checkOB(snake)
        # snakeCollidedBody = checkSelfCollision(snake)
        if snakeIsOB or loopedMoves > 100:
            fails = fails + 1
            loopedMoves = 0
            del snake
            del apple
            snake = Snake()
            #Create apple
            apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black")
            #for i in range(min(len(input_train), len(output_train))):
            if snakeIsOB:
                dir = tf.get_static_value(output_train[len(output_train)-1]) #(Bad) direction of snake in final training frame
                rightTurn = (float(dir + 1) % 4) #Model- Direction float representing bad dir + 1
                model.fit(input_train[len(input_train)-1], tf.constant([rightTurn]), epochs = 1, verbose=0) #train final input frame
            #with final output frame turned to right
            # if snakeCollidedBody:
            #     dir = tf.get_static_value(output_train[len(output_train)-1]) #(Bad) direction of snake in final training frame
            #     leftTurn = (float(dir - 1) % 4) #Model- Direction float representing bad dir + 1
            #     model.fit(input_train[len(input_train)-1], tf.constant([leftTurn]), epochs = 1, verbose=0) #train final input frame
            #with final output frame turned to left
            input_train = [] #Reset input training frames
            output_train = [] #Reset output training frames

model.save("model1.h5")
