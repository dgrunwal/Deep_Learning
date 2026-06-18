"""
Keeping Training Stable   stable_training.py
============================================
A standalone example showing the techniques that keep a network
learning smoothly: NORMALIZED inputs, MINI-BATCH training with
shuffling, and a sensible LEARNING RATE. The task is the familiar
toy problem - learn to add three numbers - so all the attention can
stay on stability, not on the problem itself. These are the same
techniques Chapter 14 uses to train the digit recognizer.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: Make some training data, then NORMALIZE it.
# -------------------------------------------------------------------
# Raw inputs span 0 to 10 - a wide range relative to the network's
# sensible internal scale, which makes training slower and jumpier.
X_raw = torch.rand(600, 3) * 10           # 600 triples, values 0..10
Y     = X_raw.sum(dim=1, keepdim=True)    # the answer: their true sum

# Normalize the inputs to a small, tidy range (here, 0.0 to 1.0) by
# dividing by their maximum. This is the by-hand version of what
# ToTensor() does for the pixels in Chapter 14.
X = X_raw / 10.0

# Bundle inputs and answers so a DataLoader can serve them in batches.
dataset = torch.utils.data.TensorDataset(X, Y)

# A DataLoader feeds the network 64 examples at a time, reshuffled
# every pass so the batches differ. batch_size and shuffle are the
# two mini-batch settings from this chapter.
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------------------------------------------------
# STEP 2: A small network. PyTorch initializes its weights sensibly
# for us - we never set them by hand.
# -------------------------------------------------------------------
class SumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 16)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SumNet()

# -------------------------------------------------------------------
# STEP 3: Loss and optimizer. lr=0.01 suits this small, quick task.
# -------------------------------------------------------------------
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------------------------------------------
# STEP 4: Train in mini-batches. The five-step loop now runs once per
# BATCH, not once per pass - many more updates, far steadier learning.
# -------------------------------------------------------------------
for epoch in range(50):
    running_loss = 0.0
    for batch_X, batch_Y in loader:          # one mini-batch at a time
        predictions = model(batch_X)         # 1. FORWARD
        loss = loss_fn(predictions, batch_Y) # 2. MEASURE
        optimizer.zero_grad()                # 3. RESET
        loss.backward()                      # 4. BACKWARD
        optimizer.step()                     # 5. UPDATE
        running_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg = running_loss / len(loader)
        print(f"Epoch {epoch + 1:2d} | Average batch loss: {avg:.4f}")
