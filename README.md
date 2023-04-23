At first, I tried implementing my NN from scratch. I decided to first test out using tensorflow in order to have a base for checking the correctness of my own implementation, particularly my backpropagation algorithm.

An obstacle was deciding which inputs the model should receive. My goal was to make great use of the NN, in processing complex inputs such as a vision grid in order to classify the optimal direction decision. However, I encountered a direct relation between the complexities of the input and the model. For example, if the model's weights are expected to have a high variance, it is less likely that the model will ultimately have a high accuracy. 

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

What I learned:

The first major mistake that I made when making this project is that I attempted to build a feedforward neural network without using a library such as tensorflow to test it. Although I was successful in implementing a multilayer network with backpropagation, using MSE as its loss function, I wasn't able to measure its effectiveness with different permutations of layer settings such as activation functions, optimizers, and loss functions without the program becoming too complex. Thus, I decided to implement the game AI using tensorflow, so that I can later apply my own CNN implementation with a reference with which to test.

Mistakes in applying the keras model:

I learned that when using CNNs, it is important to select the training model carefully. When using CNNs, it is important to notice that input data may have conflicting effects on the network's weights when training the model. To use this application as an example, if the snake's recognition of the apple contributes to a decrease in the loss but input training frames in which the snake recognized the apple are used to apply a negative Q value, this will lead to an unclear regression estimation, given that the snake recognizes the apple.

The problem that I solved using this imformation was where the snake's performance began todecrease with increasing iterations. In order to solver this, I simplified my training method by using only the last training frames instead of all frames, from the run. In other words, I trained the model using only the last frames that the snake received before hitting a wall or itself.

When training the model, the output training data must be specific to the intended action of the model.

The snake tended to loop, and after comparing the input and output of the keras model during this, I noticed that the model was predicting that the snake should turn 180 degrees. To fix this I am going to alter the direction prediction step to use the greatest value excluding the direction opposite to the snake's current direction. This was done by setting the tensorflow prediction vector's value at the index '(snake.direction + 2) % 4' to 0.0 before finding the vector's maximum argument.