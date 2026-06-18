"""
Working With Image Data       image_classifier.py
===============================================================
A tiny, self-contained image classifier. To avoid any download we
build a few small 8x8 grayscale "images" by hand:
  - "bright" images (mostly high pixel values)  -> class 1
  - "dark"   images (mostly low  pixel values)  -> class 0

We perform, in miniature, the SAME preparation MNIST needs in
Chapter 14: turn images into tensors, NORMALIZE pixels to 0..1,
FLATTEN each grid into a row, BATCH with a DataLoader, then train
a CLASSIFIER (two output scores) with Cross-Entropy Loss.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: Make some 8x8 "images" - pixel values from 0 to 255
# -------------------------------------------------------------------
# Each image is an 8x8 grid (like a tiny grayscale picture). Bright
# images have large pixel values; dark images have small ones. We
# add a little randomness so no two images are identical.
def make_image(bright):
    base = 200 if bright else 40           # mostly light or mostly dark
    img = base + 30 * torch.randn(8, 8)    # add some noise
    return img.clamp(0, 255)               # keep values in 0..255

images = []
labels = []
for _ in range(50):
    images.append(make_image(bright=True));  labels.append(1)   # class 1
    images.append(make_image(bright=False)); labels.append(0)   # class 0

X = torch.stack(images)                    # shape: (100, 8, 8)
Y = torch.tensor(labels)                   # shape: (100,)
print("Raw image batch shape:", X.shape, " pixel range:",
      f"{X.min():.0f}..{X.max():.0f}")

# -------------------------------------------------------------------
# STEP 2: NORMALIZE - rescale pixels from 0..255 down to 0.0..1.0
# -------------------------------------------------------------------
# This is feature scaling applied to pixels - the by-hand version of
# what transforms.ToTensor() does for you in Chapter 14.
X = X / 255.0
print("After normalizing,  pixel range:",
      f"{X.min():.2f}..{X.max():.2f}")

# -------------------------------------------------------------------
# STEP 3: BATCH - let a DataLoader serve the data in small groups
# -------------------------------------------------------------------
# A TensorDataset pairs each image with its label; the DataLoader
# feeds them out 16 at a time and shuffles them each pass.
dataset = torch.utils.data.TensorDataset(X, Y)
loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------------------------------------------
# STEP 4: A CLASSIFIER - it FLATTENS each image, then scores 2 classes
# -------------------------------------------------------------------
# nn.Flatten turns each 8x8 grid into a row of 64 numbers. The final
# layer outputs 2 scores (one per class); the higher score is the
# guess. This is the same shape as Chapter 14, only 2 classes not 10.
model = nn.Sequential(
    nn.Flatten(),            # 8x8 grid -> row of 64 numbers
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),        # 2 output scores: class 0 vs class 1
)

loss_fn   = nn.CrossEntropyLoss()          # the classifier loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------------------------------------------
# STEP 5: Train - the same five-step loop, now over batches
# -------------------------------------------------------------------
for epoch in range(10):
    for batch_images, batch_labels in loader:     # one batch at a time
        scores = model(batch_images)              # 1. FORWARD
        loss   = loss_fn(scores, batch_labels)    # 2. MEASURE
        optimizer.zero_grad()                     # 3. RESET
        loss.backward()                           # 4. BACKWARD
        optimizer.step()                          # 5. UPDATE
    print(f"Epoch {epoch+1:2d} | loss {loss.item():.4f}")

# -------------------------------------------------------------------
# STEP 6: Check accuracy on the whole set
# -------------------------------------------------------------------
with torch.no_grad():
    predicted = torch.argmax(model(X), dim=1)     # highest score wins
    accuracy  = (predicted == Y).float().mean().item()
print(f"\nAccuracy: {100*accuracy:.1f}%")
