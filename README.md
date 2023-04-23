# Python Tensorflow Deep Neural Network Q-Learning Snake Game Agent

## Trains a Deep Neural Network to win a game of Snake using Q-Learning

## This project is a Python application that trains a deep neural network using the game state as input and the snake's direction as output. It works by:

* Using a 3-layer Keras model to collect input describing (for each direction) 1. the snake head's location relative to the apple (4 nodes), 2. the snake head's boolean value of being next to a wall (4 nodes), 3. the snake head's boolean value of being next to its body (4 nodes), and 4. the snake's direction (1 one-hot node).
* The Keras model uses the adam optimizer, the sparse categorical crossentropy loss function, and the relu and softmax activation functions for the hidden and output layers, respectively.
* How the model is trained:
    * The model is fit using the following data when an apple is eaten: input: the last (screen width) game states seen. output: the corresponding directions during the last (screen width) game states.
    * The model is fit using the following data when the snake goes out of bounds: input: the last game state seen. output: the rightward rotation from the corresponding direction during the last game state.
    * The model is fit using the following data when the snake collides with itself: input: the last game state seen. output: the leftward rotation from the corresponding direction during the last game state.
* How the model's performance is assessed:
    The model's performance is measured as the ratio of total apples eaten to the total fails (out of bounds or self-collisions). Every 100 frames, the x value (# of iterations) and the y value (# of apples / # of fails) is added to a list of 'stats'. This list is saved to a file, 'Scores.txt' to be loaded when the program is rerun. This 'stats' list is also displayed to the user using MatplotLib, and it is refreshed every 100 frames.


## Video
(Coming soon!)


## How to use this application:
(Soon to be deployed!)

Debugging with Tensorboard:
```
    # Clear any logs from previous runs
    rm -rf ./logs/
    tensorboard --logdir logs/fit
    tensorboard --logdir logs/gradient_tape
```

## What I learned:

The first major mistake that I made when making this project is that I attempted to build a feedforward neural network without using a library such as tensorflow to test it. Although I was successful in implementing a multilayer network with backpropagation, using MSE as its loss function, I wasn't able to measure its effectiveness with different permutations of layer settings such as activation functions, optimizers, and loss functions without the program becoming too complex. Thus, I decided to implement the game AI using tensorflow, so that I can later apply my own CNN implementation with a reference with which to test.

Mistakes in applying the keras model:

First, because models take time for emergent properties to appear as bugs, I learned that it is better to start off with a simple and highly constrained model that achieves the desired purpose. The simple model can then be iterated to achieve further goals without struggling debugging a complex model over long training periods. This occurred when I assumed that the model would perform better with more input nodes; using 8 directions and 3 types of sensory input. The model was very unpredictable, took a long time to train, and after much wasted training time, begain to fail to estimate the proper function regression. I solved this issue by changing the model's input to only 4 directions of vision and simplifying the sensory input (e.g. instead of distance seen in each direction, using a boolean to represent if a particular object is seen).

I learned that when using CNNs, it is important to select the training model carefully. When using CNNs, it is important to notice that input data may have conflicting effects on the network's weights when training the model. To use this application as an example, if the snake's recognition of the apple contributes to a decrease in the loss but input training frames in which the snake recognized the apple are used to apply a negative Q value, this will lead to an unclear regression estimation, given that the snake recognizes the apple.

The problem that I solved using this imformation was where the snake's performance began todecrease with increasing iterations. In order to solver this, I simplified my training method by using only the last training frames instead of all frames, from the run. In other words, I trained the model using only the last frames that the snake received before hitting a wall or itself.

When training the model, the output training data must be specific to the intended action of the model. For example, the training output for the snake cannot be the direction vector [0, 0, 0, 0] in the case of negative reinforcement. Instead, I chose to specify a right turn when encountering walls and a left turn when encountering a body block.

The snake tended to loop, and after comparing the input and output of the keras model during this, I noticed that the model was predicting that the snake should turn 180 degrees. To fix this I altered the direction prediction step to use the greatest value excluding the direction opposite to the snake's current direction. This was done by setting the tensorflow prediction vector's value at the index '(snake.direction + 2) % 4' to 0.0 before finding the vector's maximum argument.

Another factor ccausing looping was an excess reliance on the model by the snake. Once the model learned a direction from which it ate the apple, it tended to always prefer this direction. This issue was solved by adding a random direction change every 16 steps, to prevent too much reliance on the model and allow the snake to learn to encounter the apple from all directions.

#Current Issues:
The program crashes when the Scores1.txt file becomes too large, so the initial values of the file must be erased periodically. My plan is to consolidate training stats lists into a single plot coordinate.