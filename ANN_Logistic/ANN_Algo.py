import numpy as np
import matplotlib.pyplot as plt

from ANN_Logistic.Forward_Backward_Propagation import Propagation


class ANN:
    def __init__(self):
        flag = True
        while flag:
            args = input("Please enter the number of units in each layer: ")
            layer_dim_str = args.split()
            count = 1
            for each in layer_dim_str:
                if each.isdigit():
                    count += 1
                else:
                    print("illegal input! please try again")
                    flag = False
                    break
            if count == len(layer_dim_str):
                self.layer_dims = [int(x) for x in layer_dim_str]
                flag = False

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def compute_cost(self, AL, Y):
        """
        Implement the cost function (Logistic regression).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (0 if not-survived, 1 if survived), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -1.0 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        cost = np.squeeze(cost)  # To make sure the cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
            parameters["b" + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]
        return parameters

    def train_L_Layer_Model(self, X, Y, layer_dims, learning_rate=0.003, num_iterations=5000, print_cost=False):
        """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- data, numpy array of shape (number of examples, num_input_unit)
            Y -- true "label" vector (containing 0 if not-survived, 1 if survived), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """
        np.random.seed(1)
        costs = []  # keep track of cost

        parameters = ANN.initialize_parameters_deep(layer_dims=layer_dims)  # initialize paras

        # for k, v in parameters.items():
        #     print(k)
        #     print(v.shape)

        # Loop (Gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = Propagation.L_model_forward(X, parameters=parameters)

            # Compute cost.
            cost = ANN.compute_cost(AL=AL, Y=Y)

            # Backward propagation.
            grads = Propagation.L_model_backward(AL=AL, caches=caches, Y=Y)

            # Update parameters.
            parameters = self.update_parameters(grads=grads, learning_rate=learning_rate, parameters=parameters)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = Propagation.L_model_forward(X, parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        if y is not None:  # Evaluation
            # print results
            print("predictions: " + str(p))
            print("true labels: " + str(y))
            print("Accuracy: " + str(np.sum((p == y) / m)))

        else:  # prediction for test data
            pass

        return p
