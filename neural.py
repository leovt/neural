import numpy as np

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, N0, N1, N2):
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.weights1 = np.random.rand(N0, N1)
        self.bias1 = np.random.rand(N1)
        self.weights2 = np.random.rand(N1, N2)
        self.bias2 = np.random.rand(N2)

    def apply(self, input):
        M, M0 = input.shape
        assert M0 == self.N0

        # forward
        layer1 = sigmoid(np.dot(input, self.weights1) + self.bias1)
        output = sigmoid(np.dot(layer1, self.weights2) + self.bias2)
        return output

    def train(self, input, y):
        M, M0 = input.shape
        assert M0 == self.N0
        K, K0 = y.shape
        assert K0 == self.N2
        assert K == M

        # forward
        layer1 = sigmoid(np.dot(input, self.weights1))
        output = sigmoid(np.dot(layer1, self.weights2))

        assert layer1.shape == (M, self.N1)
        assert output.shape == (M, self.N2)

        # backward
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(layer1.T, (2*(y - output) * sigmoid_derivative(output)))
        assert d_weights2.shape == self.weights2.shape == (self.N1, self.N2)
        d_bias2 = np.sum(2*(y - output) * sigmoid_derivative(output), axis=0)
        assert d_bias2.shape == self.bias2.shape == (self.N2, )

        d_weights1 = np.dot(input.T,  (np.dot(2*(y - output) * sigmoid_derivative(output), self.weights2.T) * sigmoid_derivative(layer1)))
        assert d_weights1.shape == self.weights1.shape == (self.N0, self.N1)
        d_bias1 = np.sum(np.dot(2*(y - output) * sigmoid_derivative(output), self.weights2.T) * sigmoid_derivative(layer1), axis=0)
        assert d_bias1.shape == self.bias1.shape == (self.N1, )

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.bias1 += d_bias1
        self.bias2 += d_bias2

        return output


# Each row is a training example, each column is a feature  [X1, X2, X3]
x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,0,1]), dtype=float)
y=np.array(([0],[1],[1],[0],[1]), dtype=float)

nn = NeuralNetwork(3, 4, 1)
for i in range(1500): # trains the NN 1,000 times
    output = nn.train(x,y)
    if i % 100 ==0:
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(x))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(output))
        print ("Loss: \n" + str(np.mean(np.square(y - output)))) # mean sum squared loss
        print ("\n")

print(nn.apply(np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]))))
for k in ([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]):
    print(k, nn.apply(np.array((k,))))

print(nn.weights1, nn.bias1)
print(nn.weights2, nn.bias2)
