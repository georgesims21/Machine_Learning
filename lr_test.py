#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
"""The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        
        """
        a = activation
        b = bias
        w = weight
        sigmoid : see sigmoid function
        
        Role : Loops through the whole network and updates each neurons activation using the sigmoid function
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, X_train, Y_train, X_validation, Y_validation, epochs, mini_batch_size, learning_rate, decay):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        training_data = zip(X_train, Y_train)
        validation_data = zip(X_validation, Y_validation)

    
        """Take the training data and make a list out of it"""
        training_data = list(training_data)
        
        """Check if there is data in the test_data"""
        if validation_data:
            validation_data = list(validation_data)
            n_validation_data = len(validation_data)
        
        """
        Mini-batches: Each mini-batch contains mini_batch_size elements from the training set.
        
        Splits the training data into mini-bachtes, and for each mini-batches we train the network. 
        
        """  

#       Updated for the testing
#       ========================
        mini_batches = []
        high_score = [0,0]
        for j in range(epochs):
            random.shuffle(training_data)
            for k in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if validation_data:
                new_score = self.evaluate(X_validation, Y_validation)
                if high_score[0] < new_score:
                    high_score[0] = new_score
                    high_score[1] = j + 1
            learning_rate = learning_rate * (1-decay)
        
        return high_score[0], high_score[1]
#       ========================


        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
    def update_mini_batch(self, mini_batch, learning_rate):
     
        #Building an empty networks filled with empty 0 
        updated_bias = []
        updated_weight = []
        for b in self.biases:
            updated_bias.append(np.zeros(b.shape))
        for w in self.weights:
            updated_weight.append(np.zeros(w.shape))
        
        """
        x: features of the instance
        y: label of the instance
        eta : learning rate
        *Chanche i[da=]
        """
        
        #Loops through the samples of the mini-batch, calls backprop on each sample.         
        for x, y in mini_batch:
            
            # returns the gradient of the loss function 
            loss_func_bias, loss_func_weight = self.backprop(x, y)
        
            #Updates the mini-batch bias and mini-batch weight by adding their respective loss function to the
            #current mini-batch's network
            updated_bias = [ub+lfb for ub, lfb in zip(updated_bias, loss_func_bias)]
            updated_weight = [uw+lfw for uw, lfw in zip(updated_weight, loss_func_weight)]

        
        #Updates each weight with the weights calculated in he minibach:
        #new weight= old_weight - new_weight*learning_rate
        #NOTE: can tweak the factor of correction by dividing the eta by the elements in the mini-batch
        #tmpList
        self.weights = [w-(learning_rate/len(mini_batch))*uw
                        for w, uw in zip(self.weights, updated_weight)]
            #for old_w, batch_w in zip(self.weights, updated_weight):
                #self.weights.append(old_w-(eta/len(mini_batch))*batch_w)
        
        #Updates each weight with the bias calculated in he minibach:
        #new bias= old_weight - new_bias*learning_rate
        #NOTE: can tweak the factor of correction by dividing the eta by the elements in the mini-batch
        self.biases = [b-(learning_rate/len(mini_batch))*ub
                       for b, ub in zip(self.biases, updated_bias)]
        

    def backprop(self, x, y):
        """Return a tuple ``(updated_bias, updated_weight)`` representing the
        gradient for the cost function C_x.  ``updated_bias`` and
        ``updated_weight`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        #Building an empty networks filled with empty 0 
        updated_bias = []
        for b in self.biases:
            updated_bias.append(np.zeros(b.shape))        
        
        updated_weight = []
        for w in self.weights:
            updated_weight.append(np.zeros(w.shape))
        
        # feedforward
        activation = x
        
        #list to store all the activations, layer by layer
        activation_list = [x]
        
        #list to store all the z vectors, layer by layer
        z_list = [] 
        
        # z : activation without sigmoid 
        # z_list : List of activation of each layer without sigmoid
        # activation : activation of a layer with sigmoid
        # activation_list : List of activation of each layer with sigmoid
        
        # Run through the network and record the activation with and without sigmoid and save it in  
        # z_list(without) and activation_list(with)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
            
            
        """
        Now that we ran through the network and calculated the activation we go backward through each layer 
        (from output to input) and update the bias and weight linked to the activations in order to get more 
        accurate results.
        """  
        #Apply output error formula on the last layer of the activation_list and z_list
        output_error = (activation_list[-1] - y) * sigmoid_prime(z_list[-1])
        
        #******Black box needs to be commented******
        #Backpropagate the error
        updated_bias[-1] = output_error
        updated_weight[-1] = np.dot(output_error, activation_list[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sp = sigmoid_prime(z)
            output_error = np.dot(self.weights[-l+1].transpose(), output_error) * sp
            updated_bias[-l] = output_error
            updated_weight[-l] = np.dot(output_error, activation_list[-l-1].transpose())
        return (updated_bias, updated_weight)
        #******End of Black box*******

    def evaluate(self, X_test, Y_test):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        test_data = zip(X_test, Y_test)
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
#       Updated for the testing
#       ========================
        return (sum(int(x == y) for (x, y) in test_results) / 100)
#       ========================

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    
    X_train = [np.reshape(x, (784, 1)) for x in training_data[0]]
    Y_train = [vectorized_result(y) for y in training_data[1]]
    
    X_validation = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    Y_validation = validation_data[1]
    
    X_test = [np.reshape(x, (784, 1)) for x in test_data[0]]
    Y_test = test_data[1]
    
    return (X_train, Y_train, X_validation, Y_validation,  X_test, Y_test)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



import pandas as pd
import os.path

# name the cols -- DON'T CHANGE
cols = ["# of hidden layers", "# of total neurons", "list of layers", "epoch", "mini-batch size", \
        "learning rate", "learning rate decay", "Winning epoch", "Acc Valid", "Acc Test"]
# create the data frame
df = pd.DataFrame(columns=cols)

####### Change these values if you wish to test different combinations
epochs_test = [10]
mini_batch_size_test = [10]
learning_rate_test = np.arange(0.25, 10, 0.25).tolist()
learning_rate_decay_test = [0.0001]
network_shapes = [[784, 30, 10]]
percentage = 0.0
X_train, Y_train, X_validation, Y_validation,  X_test, Y_test= load_data()

iterations = list(range(0,1))

for ite in enumerate(iterations):
    for lr_i, lr in enumerate(learning_rate_test):
        net = Network([784, 30, 10])
        acc_valid, epoch_count = net.SGD(X_train, Y_train, X_validation, Y_validation, 10, 10, lr, 0.0001)
        acc_test = net.evaluate(X_test, Y_test)

        # append to the df
        df = df.append({"# of hidden layers": 30,
                        # Change this to lamda loop when we modify the actual shape etc
                        "# of total neurons": 30,
                        "list of layers": 1,
                        "epoch" : 10,
                        "mini-batch size" : 10,
                        "learning rate" : lr,
                        "learning rate decay" : 0.0001,
                        "Winning epoch" : epoch_count,
                        "Acc Valid": acc_valid,
                        "Acc Test": acc_test},
                       ignore_index=True)
         # sanity check
        percentage += (1.0 / len(learning_rate_test)/len(iterations))
        print("Progress: ", '{0:.2f}'.format(percentage * 100), "%")

# name your file
csv_file = 'learning_rate_test_results.csv'

# write to file
'''If file exists, assume there is a header already'''
if os.path.isfile(csv_file):
    df.to_csv(csv_file, mode='a', index=False, header=False)
else:
    df.to_csv(csv_file, mode='a', index=False, header=True)
