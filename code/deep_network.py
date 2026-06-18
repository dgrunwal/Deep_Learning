"""
From Shallow to Deep   deep_network.py
======================================
A standalone example showing how a DEEP network is built - several
hidden layers stacked in sequence - and what that depth costs in
parameters. We do not train it here; we only define it, count its
learned numbers, and run a single forward pass to watch data flow
through every layer. The shape mirrors the digit recognizer you will
meet in Chapter 14.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# -------------------------------------------------------------------
# A SHALLOW network for comparison: ONE hidden layer.
# -------------------------------------------------------------------
class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)   # input  -> hidden
        self.relu   = nn.ReLU()
        self.out    = nn.Linear(128, 10)     # hidden -> 10 outputs

    def forward(self, x):
        x = self.relu(self.layer1(x))        # one hidden layer
        x = self.out(x)
        return x

# -------------------------------------------------------------------
# A DEEP network: THREE hidden layers stacked in sequence.
# Each layer transforms the output of the layer before it.
# -------------------------------------------------------------------
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)    # input   -> hidden 1
        self.layer2 = nn.Linear(128, 64)     # hidden1 -> hidden 2
        self.layer3 = nn.Linear(64, 32)      # hidden2 -> hidden 3
        self.out    = nn.Linear(32, 10)      # hidden3 -> 10 outputs
        self.relu   = nn.ReLU()

    def forward(self, x):
        # Data climbs the ladder one layer at a time. The output of
        # each layer becomes the input to the next - this chaining is
        # the whole meaning of "deep".
        x = self.relu(self.layer1(x))        # simple features
        x = self.relu(self.layer2(x))        # combined into larger parts
        x = self.relu(self.layer3(x))        # combined into richer shapes
        x = self.out(x)                      # 10 final scores
        return x

# -------------------------------------------------------------------
# Count the learned numbers (parameters) in any network.
# -------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

shallow = ShallowNet()
deep    = DeepNet()

print(f"Shallow network parameters: {count_parameters(shallow):,}")
print(f"Deep network parameters:    {count_parameters(deep):,}")

# -------------------------------------------------------------------
# One forward pass through the DEEP network, just to watch the shape
# change as data flows from 784 inputs down to 10 output scores.
# (No training - this is a pure prediction, as in Chapter 6.)
# -------------------------------------------------------------------
sample = torch.randn(1, 784)                 # one fake "image", flattened

with torch.no_grad():
    output = deep(sample)

print(f"\nInput shape:  {tuple(sample.shape)}")
print(f"Output shape: {tuple(output.shape)}  (one score per digit 0-9)")
