"""
A First Tensor  tensor_basics.py
================================
A tiny, standalone tour of the one data structure all of deep
learning runs on: the tensor. No neural network here - just the
container that every later chapter feeds into one.
"""

import torch

# -------------------------------------------------------------------
# 1. Build a tensor: 3 houses, 2 features each (size, bedrooms).
#    This is a 2-D tensor - a matrix of rows and columns.
# -------------------------------------------------------------------
houses = torch.tensor([[1400.0, 3.0],
                       [1600.0, 3.0],
                       [ 900.0, 2.0]])

# 2. Ask the tensor to describe itself.
print("Tensor:")
print(houses)
print("Shape:", houses.shape)   # (rows, columns)
print("Dims: ", houses.ndim)    # how many dimensions
print("Dtype:", houses.dtype)   # the kind of numbers it holds

# 3. Indexing: pull out one row (the first house) and one value.
print("\nFirst house:      ", houses[0])      # a 1-D vector
print("Its size only:    ", houses[0, 0])   # a single scalar

# 4. Reshaping: flatten the 3x2 grid into one long row of 6 numbers.
#    This is exactly what Chapter 14 does to a 28x28 image.
flat = houses.reshape(-1)
print("\nFlattened:        ", flat)
print("New shape:        ", flat.shape)

# 5. Scaling: divide the sizes so they sit in a small, tidy range,
#    just like Chapter 3 scaled features and Chapter 14 scales pixels.
scaled = houses / houses.max(dim=0).values
print("\nScaled to 0-1:")
print(scaled)
