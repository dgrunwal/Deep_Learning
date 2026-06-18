"""
One Neuron, From Scratch  single_neuron.py
==========================================
We build the smallest possible neural network - a single neuron - and
teach it one simple rule by example. The neuron has just one weight and
one bias to learn. By the end it can take a number it has never seen and
give a sensible answer.

This is the exact same machinery used in the MNIST digit network of
Chapter 14, shrunk down until the whole thing fits on one screen:
  - a Linear layer   (the multiply-and-add part of a neuron)
  - an activation    (ReLU, the decide part)
  - a loss function  (how wrong the neuron is)
  - an optimizer     (nudges the weight and bias to be less wrong)
  - a training loop  (the same five steps you will see again later)
"""

import torch
import torch.nn as nn

# Fix the random seed so your numbers match the book closely.
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: The Data - one simple rule to learn
# -------------------------------------------------------------------
# We want the neuron to learn the rule "double it": given x, return 2*x.
# Each input is a single number; each target is that number doubled.
inputs  = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# -------------------------------------------------------------------
# STEP 2: The Neuron
# -------------------------------------------------------------------
# nn.Linear(1, 1) is a single neuron: one input, one output. PyTorch
# creates its one weight and one bias for us. We follow it with ReLU,
# the activation that passes positives through and zeroes out negatives.
class OneNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # the multiply-and-add part
        self.relu = nn.ReLU()           # the decide part

    def forward(self, x):
        x = self.linear(x)              # weight * input + bias
        x = self.relu(x)                # keep it positive
        return x

neuron = OneNeuron()

# -------------------------------------------------------------------
# STEP 3: How to measure error and how to learn
# -------------------------------------------------------------------
# MSELoss measures how far the neuron's answers are from the targets.
# SGD is the optimizer that nudges the weight and bias to shrink that error.
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(neuron.parameters(), lr=0.01)

# -------------------------------------------------------------------
# STEP 4: Train the Neuron (the same five steps as the big network)
# -------------------------------------------------------------------
for epoch in range(200):
    predictions = neuron(inputs)            # 1. forward pass
    loss = loss_fn(predictions, targets)    # 2. measure error
    optimizer.zero_grad()                   # 3. reset old gradients
    loss.backward()                         # 4. backprop
    optimizer.step()                        # 5. update weight and bias
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d} | loss: {loss.item():.4f}")

# -------------------------------------------------------------------
# STEP 5: Inspect what the neuron learned, then use it
# -------------------------------------------------------------------
learned_weight = neuron.linear.weight.item()
learned_bias   = neuron.linear.bias.item()
print(f"\nLearned weight: {learned_weight:.3f}  (we hoped for ~2.0)")
print(f"Learned bias:   {learned_bias:.3f}  (we hoped for ~0.0)")

test_value = torch.tensor([[5.0]])
prediction = neuron(test_value).item()
print(f"\nNeuron's answer for input 5: {prediction:.2f}  (true answer: 10)")
