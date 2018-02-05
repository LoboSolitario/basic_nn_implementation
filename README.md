# basic_nn_implementation

This is a very basic implementation of a neural network using python. This program takes the data input, initializes the neural network, then finds the output from the neural network using forward propagation, calculates the error and back propagates it and finally trains the network by updating the weights. It outputs the cost after each epoch. 

In forward propagation, first the activation for each neuron in hidden layer is calculated and then it is passed to the sigmoid function (transfer function). Then the output from the first layer is considered as the input for the hidden layer and again the same technique is applied.

In back propagation, firstly the error is calculated for each neuron in the output layer and then the error is propagated backwards and then again the steps are repeated for the previous layer.

In training the network, I updated the weights according to the error calculated in the previous step and gave the output of the error after each epoch.The cost gradually decreases and shows that the network is learning
