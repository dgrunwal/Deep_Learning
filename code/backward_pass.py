"""
Backward Propagation: Learning From Mistakes   backward_pass.py
===============================================================
A minimal example that actually LEARNS. We build a tiny network
and train it, with the five-step loop, to add three numbers
together. Watch the loss fall as backward propagation nudges the
weights toward better answers, epoch after epoch.
"""

import torch
import torch.nn as nn

# Fix the random seed so your numbers closely match the book.
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: A tiny network (the same shape you met in Chapter 6)
# -------------------------------------------------------------------
# 3 inputs -> 8 hidden neurons -> ReLU -> 1 output.
class SumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 8)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = SumNet()

# -------------------------------------------------------------------
# STEP 2: Some training data - inputs and the answers we want
# -------------------------------------------------------------------
# X holds 40 rows of three random numbers (each between 0 and 10).
# Y holds the answer we want for each row: the sum of its three
# numbers. The network is NEVER told the rule "add them up" - it
# must discover that rule for itself, purely from these examples.
X = torch.rand(40, 3) * 10           # 40 triples, values in 0..10
Y = X.sum(dim=1, keepdim=True)       # the true sum of each triple

# -------------------------------------------------------------------
# STEP 3: Choose how to measure error and how to learn
# -------------------------------------------------------------------
loss_fn   = nn.MSELoss()                              # how wrong are we?
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # how we adjust

# -------------------------------------------------------------------
# STEP 4: Train - the five-step loop, run 500 times
# -------------------------------------------------------------------
for epoch in range(500):
    predictions = model(X)              # 1. FORWARD  : make a prediction
    loss = loss_fn(predictions, Y)      # 2. MEASURE  : how wrong were we?

    optimizer.zero_grad()               # 3. RESET    : clear old gradients
    loss.backward()                     # 4. BACKWARD : compute corrections
    optimizer.step()                    # 5. UPDATE   : nudge the weights

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.4f}")

# -------------------------------------------------------------------
# STEP 5: Try the trained network on a new, unseen triple
# -------------------------------------------------------------------
with torch.no_grad():
    test = torch.tensor([[3.0, 4.0, 5.0]])   # true sum is 12.0
    print("\nInput:", test)
    print("Predicted sum:", model(test).item())
