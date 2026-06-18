"""
Choosing Activation Functions   activations_demo.py
===================================================
A standalone tour of the four activations a beginner meets:
ReLU, sigmoid, tanh, and softmax. We feed ONE row of ten raw
scores - exactly the shape the Chapter 14 digit network produces
for a single image - and watch what each activation does to it.
There is no training here; we are just inspecting the functions.
"""

import torch
import torch.nn as nn

# Ten raw "scores," one per digit 0-9. These are made up, but they
# look like the unactivated output (the "logits") of a classifier:
# some negative, some positive, on no particular scale.
scores = torch.tensor([-2.0, 0.5, 1.0, 3.5, -1.0,
                        0.0, 2.0, 5.0, -0.5, 1.5])

print("Raw scores:")
print(scores)

# -------------------------------------------------------------------
# ReLU - keeps positives, zeros out negatives. Used in HIDDEN layers.
# -------------------------------------------------------------------
relu = nn.ReLU()
print("\nReLU  (max(0, x)) - negatives become 0:")
print(relu(scores))

# -------------------------------------------------------------------
# Sigmoid - squashes each number into 0..1 (an S-curve).
# Used at the OUTPUT for a single yes/no decision.
# -------------------------------------------------------------------
sigmoid = nn.Sigmoid()
print("\nSigmoid - each value squashed into 0..1:")
print(sigmoid(scores))

# -------------------------------------------------------------------
# Tanh - squashes each number into -1..+1, centered on zero.
# -------------------------------------------------------------------
tanh = nn.Tanh()
print("\nTanh - each value squashed into -1..+1:")
print(tanh(scores))

# -------------------------------------------------------------------
# Softmax - turns the WHOLE row into probabilities that sum to 1.
# Used at the OUTPUT when choosing among several classes (e.g. digits).
# dim=0 means "treat this 1-D row as the set of competing scores."
# -------------------------------------------------------------------
softmax = nn.Softmax(dim=0)
probs = softmax(scores)
print("\nSoftmax - probabilities across the 10 classes:")
print(probs)
print("They sum to:", probs.sum().item())

# The predicted class is simply the position of the largest score.
# Note: argmax on the raw scores gives the same answer as on probs,
# because softmax never changes which value is largest.
print("\nPredicted class (argmax):", torch.argmax(scores).item())
