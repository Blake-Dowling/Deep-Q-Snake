At first, I tried implementing my NN from scratch. I decided to first test out using tensorflow in order to have a base for checking the correctness of my own implementation, particularly my backpropagation algorithm.

An obstacle was deciding which inputs the model should receive. My goal was to make great use of the NN, in processing complex inputs such as a vision grid in order to classify the optimal direction decision. However, I encountered a direct relation between the complexities of the input and the model. For example, if the model's weights are expected to have a high variance, it is less likely that the model will ultimately have a high accuracy. 

