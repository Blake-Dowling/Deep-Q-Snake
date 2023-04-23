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


#look produces input to tensorlow consisting of 16 nodes: 2 nodes for each 
# of 8 directions seen by the snake's head.
# Tensorflow Input Vector:
# Objects: 0.0 = no object/wall, 1.0 = snake body, 2.0 = apple
# input[0] - object seen to right
# input[1] - object seen to bottom-right
# input[2] - object seen to bottom
# input[3] - object seen to bottom-left
# input[4] - object seen to left
# input[5] - object seen to top-left
# input[6] - object seen to top
# input[7] - object seen to top-right
# input[8] - distance of object seen to right
# input[9] - distance of object seen to bottom-right
# input[10] - distance of object seen to bottom
# input[11] - distance of object seen to bottom-left
# input[12] - distance of object seen to left
# input[13] - distance of object seen to top-left
# input[14] - distance of object seen to top
# input[15] - distance of object seen to top-right

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
                   min(WIDTH-head[0], WIDTH-head[1]),
                   WIDTH-head[1],
                   min(head[0]+1, WIDTH-head[1]),
                   head[0]+1,
                   min(head[0]+1, head[1]+1),
                   head[1]+1,
                   min(WIDTH-head[0], head[1]+1)] #Distance to wall in each direction 
    seen = [[] for i in range(8)] #Coordinates in each direction
    object = [0.0 for i in range(8)] #Object seen in each direction
    wall = [0.0 for i in range(8)] #Wall seen in each direction
    #distance = [0.0 for i in range(8)] #Distance seen in each direction

    for direction in range(0, 8): #For each direction
        for i in range(1, distToWall[direction]): #For each coordinate from head to wall in current direction
            match (direction):
                case 0: nextLoc = (head[0]+i, head[1]) #Block coordinates seen to right
                case 1: nextLoc = (head[0]+i, head[1]+i) #Block coordinates seen to bottom-right
                case 2: nextLoc = (head[0], head[1]+i) #Block coordinates seen to bottom
                case 3: nextLoc = (head[0]-i, head[1]+i) #Block coordinates seen to bottom-left
                case 4: nextLoc = (head[0]-i, head[1]) #Block coordinates seen to left
                case 5: nextLoc = (head[0]-i, head[1]-i) #Block coordinates seen to top-left
                case 6: nextLoc = (head[0], head[1]-i) #Block coordinates seen to top
                case 7: nextLoc = (head[0]+i, head[1]-i) #Block coordinates seen to top-right
                case _: break
            seen[direction].append(nextLoc) #Blocks seen so far in direction
            # for block in snake.blocks:
            #     if block.loc == nextLoc: #Last seen block was snake body
            #         object[direction] = 1.0
            #         break
            if nextLoc == apple.loc: #Last seen block was apple
                object[direction] = 1.0
                break
            # if object[direction] > 0.0: #If object seen, stop line of vision
            #     for i in range(8):
            #         if i != direction:
            #             wall[i] = 1.0
            #     break
        if len(seen[direction]) <= 0: #If vision length is 0 in direction
            wall[direction] = 1.0 #Set wall to 1 for that direction
    blocksSeen = []
    #Highlight blocks seen in each direction
    for seenDirection in seen:
        for seenBlock in seenDirection:
            blocksSeen.append(Block(seenBlock[0], seenBlock[1], "", "white"))
    #return np.array(object + distance)
    print([object + wall])
    return tf.constant([object + wall])# + distance])
    #return np.array([object.extend(distance)])



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
    model = keras.Sequential()
    layer0 = keras.layers.Flatten(input_shape=([16]))
    model.add(layer0)
    layer1 = keras.layers.Dense(16, activation="relu")
    model.add(layer1)
    layer2 = keras.layers.Dense(4, activation="softmax")
    model.add(layer2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])





    # model = keras.models.load_model("model8.h5")



    # testMatrixIn = look(snake, apple)
    # testOut = 1.0

    visualPlot1 = visual.visual.MatrixPlot(window, 0, 200)
    visualPlot2 = visual.visual.MatrixPlot(window, 0, 450)
    #print("In: ", testMatrixIn)
    

    # while True:
    #     time.sleep(.01)
    #     visualPlot1.plotMatrix(testMatrixIn)
    #     testMatrixOut = model.predict(testMatrixIn)
    #     print("Out: ", testMatrixOut)
    #     visualPlot2.plotMatrix(testMatrixOut)
    #     window.update()
    #     #model.fit(testMatrixIn, np.array([[0.0, 1.0, 0.0, 0.0]]), epochs=1, verbose=0)
    #     model.fit(testMatrixIn, tf.constant([testOut]), epochs=1, verbose=0)






    #time.sleep(30)
    #Animation loop
    TRAIN_MINUTES = 5
    loopedMoves = 0
    apples = 0
    fails = 0
    iteration = 0
    while iteration < 1000*TRAIN_MINUTES:
        iteration = iteration + 1
        loopedMoves = loopedMoves + 1
        
        time.sleep(.001)
        print("Iteration: " + str(iteration))
        
        ate = checkAte(snake, apple) #Check if head on apple
        if ate:
            apples = apples + 1
            loopedMoves = 0
            del apple 
            apple = Block(random.randint(0, WIDTH-1), random.randint(0, WIDTH-1), "red", "black") #Move apple
            for i in range(min(len(input_train), len(output_train))): #For each training frame
                dir = tf.get_static_value(output_train[i]) #(Good) direction of snake with current frame
                model.fit(input_train[i], tf.constant([dir]), epochs = 1, verbose=0) #Train each training state with direction frame
            input_train = []
            output_train = []
        
        #Append a tensorflow input based on snake's seen objects and their distances for each of eight directions
        currentVision = look(snake, apple) #Create current training frame
        input_train.append(currentVision) #Save each vision input for training (If successful)
        
        tfOutput = model.predict(currentVision, verbose=0) #Get tf model's best direction guess with current state
        visualPlot1.plotMatrix(currentVision) 
        visualPlot2.plotMatrix(tfOutput)
        print(tfOutput)
        
        bestDirection = tf.get_static_value(tf.math.argmax(tfOutput[0], output_type=tf.int64)) #Get max value of tf vector


        snake.dir = bestDirection #Set snake's new direction
        directionTensor = tf.constant(float(bestDirection)) #Training direction for current state
        output_train.append(directionTensor) #Save each chosen direction output for training (If successful)
        window.update()
        snake.move(ate) #Move snake
        snakeIsOB = checkOB(snake)
        if snakeIsOB or loopedMoves > 24:
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
            input_train = [] #Reset input training frames
            output_train = [] #Reset output training frames


    # model.save("model8.h5")
    # scoreFile = open("Scores8", "a+")
    # scoreFile.write("Steps: " + str(iteration) +\
    #                 "Apples: " + str(apples) +\
    #                 "Fails: " + str(fails) \
    #                 + "\n")
    # scoreFile.close()