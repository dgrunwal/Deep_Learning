"""
A Simple Neural Network for Housing Price Prediction house_price.py
======================================================
This example mirrors the classic "predict house price from features"
problem. We start with raw house data and train a small neural network
to map inputs (X) to a price (Y), letting the network figure out the
patterns in the middle on its own.

We use PyTorch, a popular deep learning library.
"""

import torch
import torch.nn as nn

# Fix the random seed so the network starts from the same place every
# run. This makes the numbers you see match the ones printed in the book.
# (The network's starting weights are random; without this line you would
# get slightly different results each time you run the program.)
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: The Data
# -------------------------------------------------------------------
# Each row is one house with four features:
#   [size (1000s sq ft), # bedrooms, walkability score, school score]
# These are the inputs X (just like the four inputs in the video).
X = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [1.5, 3.0, 5.0, 6.0],
    [2.0, 3.0, 7.0, 7.0],
    [2.5, 4.0, 6.0, 8.0],
    [3.0, 4.0, 8.0, 9.0],
    [3.5, 5.0, 9.0, 9.0],
], dtype=torch.float32)

# The target Y is the price (in $100,000s) we want to predict.
Y = torch.tensor([
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
], dtype=torch.float32)


# -------------------------------------------------------------------
# STEP 2: Define the Neural Network
# -------------------------------------------------------------------
# We stack simple "neurons" together, just like stacking Lego bricks.
# - Input layer: 4 features go in.
# - Hidden layer: 8 neurons that learn their own intermediate patterns
#   (the network decides what these represent, e.g. "family size").
# - Output layer: 1 neuron that outputs the predicted price.
#
# ReLU (Rectified Linear Unit) is the activation function from the video:
# it takes max(0, x), bending the line so values never go negative.
class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)   # 4 inputs -> 8 hidden neurons
        self.relu = nn.ReLU()           # the "bend the curve" function
        self.layer2 = nn.Linear(8, 1)   # 8 hidden neurons -> 1 price

    def forward(self, x):
        # Data flows forward: input -> hidden -> ReLU -> output
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = HousePriceNet()


# -------------------------------------------------------------------
# STEP 3: Choose How to Measure Error and How to Learn
# -------------------------------------------------------------------
# Loss function: Mean Squared Error measures how far off our
# predictions are from the true prices. Lower is better.
loss_fn = nn.MSELoss()

# Optimizer: this adjusts the network's internal numbers (weights)
# to reduce the loss. "lr" is the learning rate, or step size.
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


# -------------------------------------------------------------------
# STEP 4: Train the Network
# -------------------------------------------------------------------
# Training means showing the network the data many times (epochs).
# Each epoch, it predicts, measures error, and nudges its weights
# to do better next time. This is how it "figures things out by itself."
epochs = 500

for epoch in range(epochs):
    predictions = model(X)              # 1. Forward pass: make predictions
    loss = loss_fn(predictions, Y)      # 2. Measure how wrong we are

    optimizer.zero_grad()               # 3. Reset old gradients
    loss.backward()                     # 4. Backprop: compute corrections
    optimizer.step()                    # 5. Update the weights

    # Print progress every 100 epochs so we can watch it learn
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.4f}")


# -------------------------------------------------------------------
# STEP 5: Make a Prediction on a New House
# -------------------------------------------------------------------
# Now that it's trained, give the network a house it has never seen.
# size=2.2 (2200 sq ft), 3 bedrooms, walkability=6, school=7
new_house = torch.tensor([[2.2, 3.0, 6.0, 7.0]], dtype=torch.float32)

# torch.no_grad() tells PyTorch we're just predicting, not training.
with torch.no_grad():
    predicted_price = model(new_house)

print(f"\nPredicted price: ${predicted_price.item() * 100000:,.0f}")
