"""
============================================================================
 basic_deep_neural_network.py
 A Basic Deep Neural Network, built from scratch with NumPy only.
 © 2026 David Grunwald. All rights reserved.
============================================================================

WHAT THIS SCRIPT DOES
---------------------
This is a complete, runnable deep neural network written using nothing but
NumPy (no TensorFlow, no PyTorch). It builds the network the "hard way" so
you can see every moving part:

    INPUT  ->  [LINEAR -> RELU] x (L-1)  ->  LINEAR -> SIGMOID  ->  OUTPUT

Each helper function below is one building block of that pipeline. Reading
them top to bottom is like watching the network being assembled one bolt at
a time. The final section trains the network on a tiny made-up dataset so
you can watch the cost (the network's "error") go down.

HOW TO RUN IT
-------------
1. Make sure Python 3 and NumPy are installed:
       python -m pip install numpy
2. Run the script from a terminal:
       python basic_deep_neural_network.py
3. Watch the printed cost shrink over the training iterations.

There are NO external data files and NO downloads required.
============================================================================
"""

import numpy as np
import copy   # used to safely duplicate the parameter dictionary during updates


# ===========================================================================
# STEP 0 - ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# ---------------------------------------------------------------------------
# An "activation function" decides how strongly a neuron fires. A neural
# network without activation functions could only learn straight lines.
# These non-linear functions are what let it learn complex patterns.
#
# We need each activation in two directions:
#   - forward  : turn an input z into an output (used when making a prediction)
#   - backward : turn an output gradient back into an input gradient
#                (used when the network learns from its mistakes)
# ===========================================================================

def sigmoid(Z):
    """Squashes any number into the range (0, 1). Good for final yes/no output."""
    A = 1 / (1 + np.exp(-Z))
    cache = Z                       # remember Z; we'll need it for the backward pass
    return A, cache


def relu(Z):
    """ReLU = 'keep positives, zero out negatives'. Fast and works well in hidden layers."""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """Given the gradient flowing back into a sigmoid, compute the gradient before it."""
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)           # derivative of sigmoid is s*(1-s)
    return dZ


def relu_backward(dA, cache):
    """Gradient passes through where Z was positive, and is blocked where Z was negative."""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# ===========================================================================
# STEP 1 - INITIALIZE THE PARAMETERS (the network's starting "knobs")
# ---------------------------------------------------------------------------
# Every layer has two parameters:
#   W (weights) - how strongly each input matters
#   b (bias)    - a constant offset that shifts the result
#
# We start the weights as small RANDOM numbers (multiplying by 0.01 keeps them
# small) and the biases as ZEROS. Random weights are important: if every weight
# started identical, every neuron would learn the same thing.
#
# `layer_dims` is just a list of layer sizes. For example [4, 5, 3, 1] means:
#   4 inputs -> hidden layer of 5 -> hidden layer of 3 -> 1 output.
# ===========================================================================

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)               # fixed seed => same "random" numbers every run
    parameters = {}
    L = len(layer_dims)             # total number of layers (including the input layer)

    for l in range(1, L):
        # W has shape (this layer's size, previous layer's size)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        # b has shape (this layer's size, 1)
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# ===========================================================================
# STEP 2 - FORWARD PROPAGATION (making a prediction)
# ---------------------------------------------------------------------------
# Forward propagation pushes the input data through the network, layer by
# layer, until it produces a prediction. It happens in two small pieces:
#   2a. the LINEAR step:      Z = W . A_prev + b
#   2b. the ACTIVATION step:  A = relu(Z)  or  A = sigmoid(Z)
# We then chain those pieces across all layers.
# ===========================================================================

def linear_forward(A, W, b):
    """The linear step of one layer:  Z = W.A + b. Returns Z plus a cache for later."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)               # saved so backprop can reuse these values
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """One full layer: do the linear step, then apply the chosen activation."""
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Run the WHOLE forward pass:
        [LINEAR -> RELU] for every hidden layer,
        then a final LINEAR -> SIGMOID for the output.
    Returns AL (the final prediction) and a list of caches for backprop.
    """
    caches = []
    A = X                                   # the first "activation" is just the input
    L = len(parameters) // 2                # each layer contributed a W and a b

    # Hidden layers: LINEAR -> RELU, repeated L-1 times
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)

    # Output layer: LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches


# ===========================================================================
# STEP 3 - COMPUTE THE COST (how wrong is the prediction?)
# ---------------------------------------------------------------------------
# The cost is a single number summarizing the network's total error across all
# training examples. We use the cross-entropy cost, which heavily punishes
# confident-but-wrong predictions. Training = making this number as small as
# possible.
# ===========================================================================

def compute_cost(AL, Y):
    m = Y.shape[1]                                          # number of examples
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)                                 # turn [[0.4]] into 0.4
    return cost


# ===========================================================================
# STEP 4 - BACKWARD PROPAGATION (learning from the error)
# ---------------------------------------------------------------------------
# Backprop is forward prop run in reverse. It uses calculus (the chain rule)
# to figure out, for every weight and bias, "if I nudge you, how much does the
# cost change?" Those answers are the GRADIENTS (dW, db). They tell us which
# direction to push each parameter to reduce the error.
# ===========================================================================

def linear_backward(dZ, cache):
    """Given dZ for a layer, compute the gradients dW, db, and dA_prev."""
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """Undo one layer: push the gradient back through the activation, then the linear step."""
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Run the WHOLE backward pass and collect every gradient into a dictionary.
    We start at the output (sigmoid) layer, then walk backward through the
    hidden (relu) layers.
    """
    grads = {}
    L = len(caches)                  # number of layers
    Y = Y.reshape(AL.shape)          # make sure Y lines up with AL

    # The very first gradient: derivative of the cost with respect to AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Output layer (SIGMOID)
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Hidden layers (RELU), walked from the last one back to the first
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# ===========================================================================
# STEP 5 - UPDATE THE PARAMETERS (gradient descent)
# ---------------------------------------------------------------------------
# Now we actually adjust the knobs. The rule is simple:
#       new_value = old_value - learning_rate * gradient
# The learning_rate controls step size: too big and we overshoot, too small
# and learning crawls. We nudge every W and b a little, then repeat the whole
# forward/cost/backward/update loop many times.
# ===========================================================================

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)   # don't modify the caller's dictionary
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# ===========================================================================
# STEP 6 - PUT IT ALL TOGETHER: THE TRAINING LOOP
# ---------------------------------------------------------------------------
# This function ties every piece above into the classic training cycle:
#       forward  ->  cost  ->  backward  ->  update     (repeat)
# Each pass through the loop is one "iteration". Over many iterations the
# cost should steadily fall as the network gets better at the task.
# ===========================================================================

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True):
    np.random.seed(1)
    costs = []

    # Start with random weights / zero biases
    parameters = initialize_parameters_deep(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)          # 1. predict
        cost = compute_cost(AL, Y)                           # 2. measure error
        grads = L_model_backward(AL, Y, caches)              # 3. find gradients
        parameters = update_parameters(parameters, grads, learning_rate)  # 4. learn

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i:4d}: {cost:.6f}")
            costs.append(cost)

    return parameters, costs


# ===========================================================================
# STEP 7 - A TINY DEMO YOU CAN ACTUALLY RUN
# ---------------------------------------------------------------------------
# We invent a small dataset so the script does something visible. There IS a
# real (but hidden) rule baked into the labels, so the network has something
# genuine to discover. As it learns, the cost will fall and the accuracy will
# climb - proof that all the machinery above is wired together correctly.
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Basic Deep Neural Network - training demo")
    print("=" * 60)

    np.random.seed(1)

    # Make up a toy dataset with a LEARNABLE pattern:
    #   X = 4 features per example, 200 examples  -> shape (4, 200)
    #   Y = 1 when feature0 + feature1 > feature2 + feature3, else 0
    # The network does not know this rule; it must learn it from examples.
    n_features = 4
    n_examples = 200
    X = np.random.randn(n_features, n_examples)
    Y = ((X[0, :] + X[1, :]) > (X[2, :] + X[3, :])).astype(int).reshape(1, n_examples)

    # Network shape: 4 inputs -> 5 -> 3 -> 1 output
    layer_dims = [n_features, 5, 3, 1]

    print(f"\nInput  X shape: {X.shape}  (features x examples)")
    print(f"Labels Y shape: {Y.shape}  (1 x examples)")
    print(f"Network layers: {layer_dims}\n")
    print("Training... (watch the cost fall)\n")

    parameters, costs = L_layer_model(
        X, Y, layer_dims, learning_rate=0.1, num_iterations=2500, print_cost=True)

    # After training, run one more forward pass and turn probabilities into 0/1
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    accuracy = np.mean(predictions == Y) * 100

    print(f"\nFinal training accuracy on the toy data: {accuracy:.1f}%")
    print("\nDone. Every function above ran as part of this training loop.")
