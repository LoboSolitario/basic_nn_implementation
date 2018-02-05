# this is a vanilla implementation of neural network
from math import exp
from random import seed , randrange, random

# initializing a neural network
# the network is initailzed as a list of layers where
# each layer is defined as a dictionary with the following
# properties: 1)weights 2)output 3)delta
def initialize(n_input , n_hidden , n_output):
    network = list()
    h_layer = [{'weights' : [random() for i in range(n_input+1)]} for i in range(n_hidden)]
    network.append(h_layer)
    o_layer = [{'weights' : [random() for i in range(n_hidden+1)]} for i in range(n_output)]
    network.append(o_layer)
    return network

# defining the transfer function : sigmoid
def sigmoid(x):
    return 1.0 / ( 1.0 + exp(-x))
#derivative of sigmoid function
def sigmoid_derivative(x):
    return x*(1.0-x)

#activating the neuron
def activate_neuron(weights,inputs):
    #this is for the bias weight as we assume x for bias as 1
    activation = weights[-1]
    #act = sum(weight*x)
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

#this is for progating the input forward in the network
def forward_propagate(network , row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            #finding activation for each neuron
            activation = activate_neuron(neuron['weights'], inputs)
            #finding sigmoid for each activated neuron
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backpropagate_error(network , expected):
    #traversing the layer from end... 
    for i in reversed(range(len(network))):
        layer = network[i]
        #new list for storing error values of each node
        errors = list()
        #if the layer is the last layer, we just subtract the given output from the expected output
        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        else:
            for j in range(len(layer)):
                error = 0
                #otherwise we find the error due to neuron in layer l+1
                #and add all the errors
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta']) 
                errors.append(error)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
            
#this updates the weights after backpropogation
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        #inputs is all the elements of the row except the last element
        inputs = row[:-1]
        #if we are considering the layer other than input layer we take input as the output of the neurons from previous layer
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        #we increment the weights
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate*neuron['delta']*inputs[j]
            neuron['weights'][-1] += l_rate*neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backpropagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
#counting the number of unique outputs
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize(n_inputs, 3, n_outputs)
train_network(network, dataset,4, 20, n_outputs)
for layer in network:
	print(layer)
                  