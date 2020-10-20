import numpy as np

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

        self.weights = [np.random.rand(N0, N1) for N0, N1 in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.rand(N1) for N1 in layers[1:]]

    def apply(self, input):
        M, M0 = input.shape
        assert M0 == self.layers[0]

        # forward
        activation = input
        for (w, b) in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(activation, w) + b)
        return activation

    def train(self, input, y):
        M, M0 = input.shape
        assert M0 == self.layers[0]
        K, K0 = y.shape
        assert K0 == self.layers[-1]
        assert K == M

        # forward
        activations = [input]
        for (w, b) in zip(self.weights, self.biases):
            activations.append(sigmoid(np.dot(activations[-1], w) + b))

        output = activations[-1]
        # backward
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_cost = 2*(y - output)

        for i, (w, b, layer_in, layer_out) in list(enumerate(zip(self.weights, self.biases, activations[:-1], activations[1:])))[::-1]:
            d_w = np.dot(layer_in.T, d_cost * sigmoid_derivative(layer_out))
            d_b = np.sum(            d_cost * sigmoid_derivative(layer_out), axis=0)

            d_cost = np.dot(d_cost * sigmoid_derivative(layer_out), w.T)

            self.weights[i] += d_w
            self.biases[i] += d_b

        return output

def main():
    # Each row is a training example, each column is a feature  [X1, X2, X3]
    x=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                [1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=float)
    y=np.array([[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1]], dtype=float)

    nn = NeuralNetwork([3, 8])
    np.set_printoptions(precision=3, suppress=True)
    for i in range(500):
        output = nn.train(x,y)
        if i%100==99:
            print (i+1, "Loss: ", np.mean(np.square(y - output)))

    print ("Input : \n" + str(x))
    print ("Actual Output: \n" + str(y))
    print ("Predicted Output: \n" + str(output))
    print ("Loss: \n" + str(np.mean(np.square(y - output)))) # mean sum squared loss
    print ("\n")

if __name__ == '__main__':
    main()
