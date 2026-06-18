"""
Recognizing Handwritten Digits  mnist_digits.py
===============================
This is the capstone project. We build, train, and evaluate a complete
deep neural network on the MNIST dataset - 70,000 small grayscale images
of handwritten digits (0 through 9). By the end the network can look at a
28x28 picture of a digit and tell you which number it is, and we inspect
the mistakes it makes.

Everything you learned earlier comes together here:
  - tensors        (the images and labels are tensors)
  - layers         (Linear layers stacked into a network)
  - activations    (ReLU between layers)
  - training loop  (forward pass, loss, backward pass, update)
  - honest evaluation (we test on images the network never trained on)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Fix the random seed so your results closely match the book.
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: Load the Data
# -------------------------------------------------------------------
# transforms.ToTensor() turns each image into a tensor and scales its
# pixel values from 0-255 down to 0.0-1.0 (a simple form of feature
# scaling, exactly like Chapter 3, but for pixels).
transform = transforms.ToTensor()

# MNIST ships as two ready-made sets: 60,000 training images and 10,000
# test images. We never train on the test set - it is our honest exam.
train_data = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# A DataLoader feeds the network the data in small batches of 64 images
# at a time, rather than all 60,000 at once. shuffle=True mixes the
# training images each epoch so the network does not learn their order.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=1000, shuffle=False)

print(f"Training images: {len(train_data)}")
print(f"Test images:     {len(test_data)}")


# -------------------------------------------------------------------
# STEP 2: Define the Neural Network
# -------------------------------------------------------------------
# Each image is 28x28 = 784 pixels. We flatten that grid into a single
# row of 784 numbers and feed it through:
#   784 inputs -> 128 hidden neurons -> ReLU
#               -> 64  hidden neurons -> ReLU
#               -> 10  outputs (one score per digit 0-9)
# The output with the highest score is the network's guess.
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()        # 28x28 image -> 784-length row
        self.layer1 = nn.Linear(784, 128)  # input  -> first hidden layer
        self.layer2 = nn.Linear(128, 64)   # hidden -> second hidden layer
        self.layer3 = nn.Linear(64, 10)    # hidden -> 10 digit scores
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)                 # raw scores (no activation here)
        return x


model = DigitNet()


# -------------------------------------------------------------------
# STEP 3: Choose How to Measure Error and How to Learn
# -------------------------------------------------------------------
# CrossEntropyLoss is the standard loss for classification. It compares
# the network's 10 scores against the true digit and gives a single
# number that is small when the network is confident and correct.
loss_fn = nn.CrossEntropyLoss()

# Adam adjusts the network's weights to reduce the loss, just as before.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -------------------------------------------------------------------
# STEP 4: Train the Network
# -------------------------------------------------------------------
# One epoch = one full pass over all 60,000 training images. We do 3.
epochs = 3

for epoch in range(epochs):
    model.train()                          # put the model in training mode
    running_loss = 0.0
    for images, labels in train_loader:    # one batch of 64 at a time
        predictions = model(images)        # 1. forward pass
        loss = loss_fn(predictions, labels)# 2. measure error
        optimizer.zero_grad()              # 3. reset old gradients
        loss.backward()                    # 4. backprop
        optimizer.step()                   # 5. update weights
        running_loss += loss.item()
    avg = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Average training loss: {avg:.4f}")


# -------------------------------------------------------------------
# STEP 5: Evaluate on the Test Set (images never seen in training)
# -------------------------------------------------------------------
model.eval()                               # put the model in evaluation mode
correct = 0
total = 0
with torch.no_grad():                      # no learning, just predicting
    for images, labels in test_loader:
        outputs = model(images)
        # For each image, the predicted digit is the index of the
        # highest of the 10 scores.
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%  ({correct} of {total} correct)")


# -------------------------------------------------------------------
# STEP 6: Inspect the Mistakes
# -------------------------------------------------------------------
# Honest evaluation means looking at what the network got wrong, not
# just the score. We collect a few misclassified test images and report
# what the true digit was versus what the network guessed.
mistakes = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        wrong = predicted != labels
        for i in range(len(images)):
            if wrong[i] and len(mistakes) < 5:
                mistakes.append((labels[i].item(), predicted[i].item()))
        if len(mistakes) >= 5:
            break

print("\nA few mistakes the network made:")
for true_digit, guessed in mistakes:
    print(f"  true digit was {true_digit}, network guessed {guessed}")
