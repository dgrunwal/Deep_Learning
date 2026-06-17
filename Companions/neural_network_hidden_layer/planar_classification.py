"""
Planar Data Classification with One Hidden Layer
================================================
A complete, self-contained neural network with a single hidden layer,
built from scratch with NumPy. It classifies a "flower"-shaped dataset
that a straight line cannot separate, demonstrating why a hidden layer
matters.

This file does NOT need the Coursera course files (planar_utils,
testCases, public_tests). It generates the dataset itself and runs
end to end. Just run:

    python planar_classification.py

Prerequisites:
    Python 3.8+ and two libraries:  numpy  and  scikit-learn
    Install them with:
        pip install numpy scikit-learn
    (scikit-learn is used only for the logistic-regression comparison.)
"""

import numpy as np
import copy
import sklearn.linear_model


# ===================================================================
# Generate the planar "flower" dataset
# (This replaces the course's load_planar_dataset() so the file is
#  self-contained. It builds two interleaved petals of red/blue points.)
# ===================================================================
def load_planar_dataset():
    np.random.seed(1)
    m = 400                      # total number of examples
    N = int(m / 2)               # points per class
    D = 2                        # two features (x1, x2)
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4                        # maximum petal radius
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    # Convention: features as rows, examples as columns.
    X = X.T
    Y = Y.T
    return X, Y


# ===================================================================
# Sigmoid activation (the course imports this; we define it here)
# ===================================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ===================================================================
# Step 1: Define the network structure (layer sizes)
# ===================================================================
def layer_sizes(X, Y):
    """
    Returns the size of the input, hidden, and output layers.
    n_x and n_y are read from the data; n_h is a design choice.
    """
    n_x = X.shape[0]   # number of input features (rows of X)
    n_h = 4            # hidden layer size - chosen by us, not the data
    n_y = Y.shape[0]   # number of outputs (rows of Y)
    return (n_x, n_h, n_y)


# ===================================================================
# Step 2: Initialize parameters (small random weights, zero biases)
# ===================================================================
def initialize_parameters(n_x, n_h, n_y):
    """
    Weights start small and RANDOM (to break symmetry so neurons learn
    different things). Biases start at zero.
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# ===================================================================
# Step 3a: Forward propagation (push data through to a prediction)
# ===================================================================
def forward_propagation(X, parameters):
    """
    Computes the prediction A2, and caches the intermediate values
    that backward propagation will need.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1     # linear step, hidden layer
    A1 = np.tanh(Z1)            # tanh activation, hidden layer
    Z2 = np.dot(W2, A1) + b2    # linear step, output layer
    A2 = sigmoid(Z2)            # sigmoid activation -> probability

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ===================================================================
# Step 3b: Compute the cost (cross-entropy: how wrong we are)
# ===================================================================
def compute_cost(A2, Y):
    """
    Cross-entropy cost, averaged over all m examples.
    Small when predictions are confident and correct.
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))  # turn [[value]] into a plain number
    return cost


# ===================================================================
# Step 3c: Backward propagation (find the gradients)
# ===================================================================
def backward_propagation(parameters, cache, X, Y):
    """
    Works backward from the output error to find how each weight and
    bias should change to reduce the cost.
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y                                      # output error
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))   # tanh derivative = 1 - A1^2
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# ===================================================================
# Step 3d: Update parameters (take one gradient-descent step downhill)
# ===================================================================
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Nudge every parameter a little in the direction that lowers the cost.
    learning_rate controls how big each step is.
    """
    W1 = copy.deepcopy(parameters["W1"])
    b1 = parameters["b1"]
    W2 = copy.deepcopy(parameters["W2"])
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# ===================================================================
# Step 4: Assemble everything into the training loop
# ===================================================================
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    The full model: initialize once, then loop the four learning steps
    (forward -> cost -> backward -> update) many times.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)        # predict
        cost = compute_cost(A2, Y)                            # measure
        grads = backward_propagation(parameters, cache, X, Y) # gradients
        parameters = update_parameters(parameters, grads)     # step downhill

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# ===================================================================
# Step 5: Predict (one forward pass, then threshold at 0.5)
# ===================================================================
def predict(parameters, X):
    """
    Use the learned parameters to classify each example as 0 or 1.
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


# ===================================================================
# Run the whole thing
# ===================================================================
if __name__ == "__main__":

    # Load the flower-shaped data
    X, Y = load_planar_dataset()
    print("Shape of X:", X.shape, " (2 features, 400 examples)")
    print("Shape of Y:", Y.shape, " (1 label,   400 examples)")
    print("Training examples: m =", X.shape[1])

    # --- Baseline: logistic regression (a single straight-line boundary) ---
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.ravel())
    LR_preds = clf.predict(X.T)                    # shape (400,)
    y_flat = Y.ravel()                             # shape (400,)
    lr_acc = np.mean(LR_preds == y_flat) * 100
    print("\nLogistic regression accuracy: %d%%" % lr_acc,
          "(a straight line cannot separate a flower)")

    # --- Neural network with one hidden layer ---
    print("\nTraining a neural network with a hidden layer of size 4...")
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

    predictions = predict(parameters, X)
    nn_acc = np.mean(predictions.ravel() == Y.ravel()) * 100
    print("\nNeural network accuracy: %d%%" % nn_acc,
          "(the hidden layer learns a curved boundary)")

    # --- Bonus: how hidden-layer size affects accuracy ---
    print("\nEffect of hidden layer size (n_h):")
    for n_h in [1, 2, 4, 5, 20, 50]:
        params = nn_model(X, Y, n_h, num_iterations=5000, print_cost=False)
        preds = predict(params, X)
        acc = np.mean(preds.ravel() == Y.ravel()) * 100
        print("  n_h = %2d  ->  accuracy %.1f%%" % (n_h, acc))
