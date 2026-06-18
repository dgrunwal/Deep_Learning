"""
A Simple Neural Network for Housing Price Prediction (Version 2)house_price_v2.py
=================================================================
This is the Chapter 3 version of the house-price predictor. It adds the
two habits every practitioner relies on:

  1. Feature scaling   - putting every input feature on a comparable range
                          so the network is not unfairly swayed by whichever
                          feature happens to have the biggest numbers.
  2. Train/test split  - holding back some houses the network never trains
                          on, so we can measure how well it truly generalizes
                          instead of how well it memorized.

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
# NEW: Split the data into a training set and a test set
# -------------------------------------------------------------------
# The first four houses are used for learning; the last two are held
# back and only used at the very end to grade the model on houses it
# has never seen.
X_train, Y_train = X[:4], Y[:4]
X_test,  Y_test  = X[4:], Y[4:]


# -------------------------------------------------------------------
# NEW: Scale the features using the TRAINING data's statistics
# -------------------------------------------------------------------
# Subtract the mean and divide by the standard deviation so every
# feature is roughly centered on zero and stretched to a similar width.
# Crucial detail: we compute the mean and std from the TRAINING houses
# only, then apply those same numbers to the test houses. The test set
# must stay a true unknown.
mean = X_train.mean(dim=0)
std  = X_train.std(dim=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std   # reuse TRAIN stats, never test stats


# -------------------------------------------------------------------
# STEP 2: Define the Neural Network
# -------------------------------------------------------------------
# Same network as before:
# - Input layer: 4 features go in.
# - Hidden layer: 8 neurons that learn their own intermediate patterns.
# - Output layer: 1 neuron that outputs the predicted price.
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
# STEP 4: Train the Network (on the scaled TRAINING data only)
# -------------------------------------------------------------------
epochs = 500

for epoch in range(epochs):
    predictions = model(X_train_scaled)       # 1. Forward pass on training data
    loss = loss_fn(predictions, Y_train)      # 2. Measure how wrong we are

    optimizer.zero_grad()                     # 3. Reset old gradients
    loss.backward()                           # 4. Backprop: compute corrections
    optimizer.step()                          # 5. Update the weights

    # Print progress every 100 epochs so we can watch it learn
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Training Loss: {loss.item():.4f}")


# -------------------------------------------------------------------
# STEP 5: Evaluate on the held-back test houses (never seen in training)
# -------------------------------------------------------------------
# torch.no_grad() tells PyTorch we're just predicting, not training.
with torch.no_grad():
    test_predictions = model(X_test_scaled)
    test_loss = loss_fn(test_predictions, Y_test)

print(f"\nTest Loss: {test_loss.item():.4f}")
for i in range(len(X_test)):
    actual    = Y_test[i].item() * 100000
    predicted = test_predictions[i].item() * 100000
    print(f"House {i+1}: Actual ${actual:,.0f} | Predicted ${predicted:,.0f}")
