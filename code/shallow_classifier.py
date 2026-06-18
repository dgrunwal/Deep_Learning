"""
A Network With One Hidden Layer   shallow_classifier.py
=======================================================
We build, train, and evaluate a SHALLOW network - one hidden
layer - on a two-class problem that a single straight line
cannot solve. This is the same shape of network and the same
classification setup used by the MNIST digit recognizer in
Chapter 14, kept small enough to read in one sitting.
"""

import torch
import torch.nn as nn

# Reproducible results: same data and same starting weights each run.
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: Make a small two-class dataset a straight line CANNOT split
# -------------------------------------------------------------------
# Two interleaved clusters (an XOR-like pattern). Each point has 2
# features (x, y). Each label is 0 or 1 - the class the point belongs to.
X = torch.tensor([
    [0.1, 0.1], [0.2, 0.3], [0.3, 0.2], [0.15, 0.25],   # cluster -> class 0
    [0.8, 0.8], [0.9, 0.7], [0.7, 0.9], [0.85, 0.75],   # cluster -> class 0
    [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.15, 0.85],   # cluster -> class 1
    [0.8, 0.1], [0.9, 0.2], [0.7, 0.3], [0.85, 0.15],   # cluster -> class 1
])
y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1])

# -------------------------------------------------------------------
# STEP 2: Define a shallow network: 2 inputs -> hidden layer -> 2 scores
# -------------------------------------------------------------------
# The hidden layer is what lets the network bend its decision boundary.
# Remove it (or its ReLU) and the network can only draw a straight line.
class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)   # 2 inputs  -> 8 hidden neurons
        self.relu   = nn.ReLU()         # the "bend": makes curves possible
        self.output = nn.Linear(8, 2)   # 8 hidden  -> 2 class scores

    def forward(self, x):
        x = self.relu(self.hidden(x))   # hidden layer + activation
        x = self.output(x)              # raw scores, one per class
        return x                        # NO softmax - the loss adds it

model = ShallowNet()

# -------------------------------------------------------------------
# STEP 3: Choose the loss and the optimizer
# -------------------------------------------------------------------
# CrossEntropyLoss is the standard loss for classification. It takes the
# raw scores and the true labels and returns a single number to minimize.
loss_fn = nn.CrossEntropyLoss()

# Adam adjusts the weights to reduce the loss, exactly as in earlier chapters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# -------------------------------------------------------------------
# STEP 4: Train - the same five-step loop, now recording the loss
# -------------------------------------------------------------------
epochs = 200
history = []                            # we save the loss each epoch

for epoch in range(epochs):
    scores = model(X)                   # 1. forward pass
    loss   = loss_fn(scores, y)         # 2. measure error
    optimizer.zero_grad()               # 3. reset old gradients
    loss.backward()                     # 4. backprop
    optimizer.step()                    # 5. update weights

    history.append(loss.item())         # record for the training curve
    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f}")

# -------------------------------------------------------------------
# STEP 5: Evaluate - did it learn the pattern?
# -------------------------------------------------------------------
with torch.no_grad():                   # predicting, not training
    scores  = model(X)
    guesses = torch.argmax(scores, dim=1)   # highest score = predicted class
    correct = (guesses == y).sum().item()

print(f"\nAccuracy: {correct} of {len(y)} correct "
      f"({100 * correct / len(y):.0f}%)")

# -------------------------------------------------------------------
# STEP 6: Read the training curve (no plotting library needed)
# -------------------------------------------------------------------
# We print the loss at a few checkpoints. A healthy curve falls quickly
# and then flattens out near zero - the sign that learning worked.
print("\nTraining curve (loss should fall and flatten):")
for e in (0, 39, 79, 119, 159, 199):
    bar = "#" * int(history[e] * 40)    # a tiny text bar chart
    print(f"  epoch {e + 1:3d} | loss {history[e]:.4f} | {bar}")
