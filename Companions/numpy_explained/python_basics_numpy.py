"""
Python Basics with NumPy
========================
A clean, runnable version of the "Python Basics with NumPy" exercises:
sigmoid, its derivative, image-to-vector reshaping, row normalization,
softmax, a vectorization speed comparison, and the L1 / L2 loss functions.

This file is self-contained. Just run it:

    python python_basics_numpy.py

It prints the result of each function so you can see them work.


C:\python3_13>python python_basics_numpy.py
1. basic_sigmoid(1) = 0.7310585786300049

   np.exp([1, 2, 3]) = [ 2.71828183  7.3890561  20.08553692]
   [1, 2, 3] + 3     = [4 5 6]

2. sigmoid(t_x)            = [0.73105858 0.88079708 0.95257413]
3. sigmoid_derivative(t_x) = [0.19661193 0.10499359 0.04517666]

4. image2vector(t_image) has shape (18, 1)

5. normalize_rows(x) =
 [[0.         0.6        0.8       ]
 [0.13736056 0.82416338 0.54944226]]

6. softmax(t_x2) =
 [[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
  1.21052389e-04]
 [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
  8.01252314e-04]]

7. L1 loss = 1.1
8. L2 loss = 0.43

--- Vectorization speed comparison ---
loop  dot = 278   (time 0.0000 ms)
numpy dot = 278   (time 0.0000 ms)

C:\python3_13>


"""

import math
import time
import numpy as np


# ===================================================================
# 1. basic_sigmoid - the sigmoid of a single number, using math.exp
# ===================================================================
def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s


# ===================================================================
# 2. sigmoid - the sigmoid of a scalar OR a whole numpy array
# ===================================================================
def sigmoid(x):
    """
    Compute the sigmoid of x.

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    # np.exp works element-wise, so this handles an entire array at once.
    s = 1 / (1 + np.exp(-x))
    return s


# ===================================================================
# 3. sigmoid_derivative - the gradient (slope) of the sigmoid
# ===================================================================
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the
    sigmoid function with respect to its input x.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- The computed gradient.
    """
    # Step 1: the sigmoid itself. Step 2: the derivative s * (1 - s).
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


# ===================================================================
# 4. image2vector - flatten a (length, height, depth) image to (N, 1)
# ===================================================================
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length * height * depth, 1)
    """
    # Read the dimensions from the image itself so this works for any size.
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v


# ===================================================================
# 5. normalize_rows - rescale each row of a matrix to unit length
# ===================================================================
def normalize_rows(x):
    """
    Normalize each row of the matrix x (so each row has unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix.
    """
    # x_norm holds the length of each row. axis=1 measures across each row;
    # keepdims=True keeps it as a column so the division broadcasts cleanly.
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x


# ===================================================================
# 6. softmax - turn each row of scores into probabilities summing to 1
# ===================================================================
def softmax(x):
    """
    Calculate the softmax for each row of the input x.
    Works for a row vector and also for matrices of shape (m, n).

    Argument:
    x -- A numpy matrix of shape (m, n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m, n)
    """
    # Exponentiate every score (makes them positive), sum each row,
    # then divide each score by its row's total. Broadcasting lines
    # the (m, 1) column of sums up against the (m, n) matrix.
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


# ===================================================================
# 7. L1 loss - sum of absolute differences between yhat and y
# ===================================================================
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y    -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function
    """
    loss = np.sum(np.abs(y - yhat))
    return loss


# ===================================================================
# 8. L2 loss - sum of squared differences between yhat and y
# ===================================================================
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y    -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function
    """
    loss = np.sum((y - yhat) ** 2)
    return loss


# ===================================================================
# Demonstration: run each function and print the result
# ===================================================================
if __name__ == "__main__":

    print("1. basic_sigmoid(1) =", basic_sigmoid(1))

    # Why we prefer numpy over math in deep learning: math.exp() cannot
    # handle an array, but np.exp() applies to every element at once.
    print("\n   np.exp([1, 2, 3]) =", np.exp(np.array([1, 2, 3])))
    print("   [1, 2, 3] + 3     =", np.array([1, 2, 3]) + 3)

    t_x = np.array([1, 2, 3])
    print("\n2. sigmoid(t_x)            =", sigmoid(t_x))
    print("3. sigmoid_derivative(t_x) =", sigmoid_derivative(t_x))

    # A 3 x 3 x 2 array stands in for an image (real images are often
    # (height, width, 3) where 3 is the red/green/blue channels).
    t_image = np.array([[[0.67826139, 0.29380381],
                         [0.90714982, 0.52835647],
                         [0.42152510, 0.45017551]],

                        [[0.92814219, 0.96677647],
                         [0.85304703, 0.52351845],
                         [0.19981397, 0.27417313]],

                        [[0.60659855, 0.00533165],
                         [0.10820313, 0.49978937],
                         [0.34144279, 0.94630077]]])
    print("\n4. image2vector(t_image) has shape", image2vector(t_image).shape)

    x = np.array([[0., 3., 4.],
                  [1., 6., 4.]])
    print("\n5. normalize_rows(x) =\n", normalize_rows(x))

    t_x2 = np.array([[9, 2, 5, 0, 0],
                     [7, 5, 0, 0, 0]])
    print("\n6. softmax(t_x2) =\n", softmax(t_x2))

    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("\n7. L1 loss =", L1(yhat, y))
    print("8. L2 loss =", L2(yhat, y))

    # -------------------------------------------------------------------
    # Vectorization: the same math, looped vs. vectorized, timed.
    # The vectorized NumPy versions are far faster on large data.
    # -------------------------------------------------------------------
    print("\n--- Vectorization speed comparison ---")
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    # Classic Python loop for the dot product
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    print(f"loop  dot = {dot}   (time {1000 * (toc - tic):.4f} ms)")

    # Vectorized dot product
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    print(f"numpy dot = {dot}   (time {1000 * (toc - tic):.4f} ms)")
