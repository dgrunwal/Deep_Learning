"""
Overfitting and Regularization      overfit_demo.py
===============================================================
A small, self-contained example that makes overfitting visible.
We train two networks on the SAME small, noisy dataset:
  model_plain    - no regularization (it will overfit)
  model_reg      - with dropout + weight decay (it generalizes)

Watch the validation loss: for the plain model it falls, then
rises again (overfitting). For the regularized model it stays
low. Same five-step training loop you already know from earlier
chapters - only now we also measure a validation set each epoch.
"""

import torch
import torch.nn as nn

# Fix the random seed so your numbers closely match the book.
torch.manual_seed(42)

# -------------------------------------------------------------------
# STEP 1: A small, noisy dataset, split THREE ways
# -------------------------------------------------------------------
# 200 samples, each with 20 input features. The "true" rule is a
# simple weighted sum, but we add random noise so a too-eager model
# can memorize the noise instead of learning the rule.
N, F = 200, 20
X = torch.randn(N, F)
true_w = torch.randn(F, 1)
Y = X @ true_w + 0.3 * torch.randn(N, 1)        # signal + noise

# Three-way split: 120 train / 40 validation / 40 test
X_train, Y_train = X[:120],     Y[:120]
X_val,   Y_val   = X[120:160],  Y[120:160]
X_test,  Y_test  = X[160:],     Y[160:]

# -------------------------------------------------------------------
# STEP 2: Two networks - one plain, one regularized
# -------------------------------------------------------------------
# Both are deliberately oversized for this tiny dataset, which is
# exactly the setup in which overfitting shows up most clearly.
def make_model(use_dropout):
    layers = [nn.Linear(F, 128), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))          # drop 50% of neurons
    layers += [nn.Linear(128, 64), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)

model_plain = make_model(use_dropout=False)
model_reg   = make_model(use_dropout=True)

loss_fn = nn.MSELoss()

# The PLAIN optimizer has no weight decay; the REGULARIZED one does.
opt_plain = torch.optim.Adam(model_plain.parameters(), lr=0.01)
opt_reg   = torch.optim.Adam(model_reg.parameters(),   lr=0.01,
                             weight_decay=1e-2)          # L2 penalty

# -------------------------------------------------------------------
# STEP 3: A training routine that also tracks the VALIDATION loss
# -------------------------------------------------------------------
def train(model, optimizer, epochs=300):
    best_val = float("inf")          # for early stopping
    for epoch in range(epochs):
        model.train()                          # dropout ON
        pred = model(X_train)                  # 1. FORWARD
        loss = loss_fn(pred, Y_train)          # 2. MEASURE (train)
        optimizer.zero_grad()                  # 3. RESET
        loss.backward()                        # 4. BACKWARD
        optimizer.step()                       # 5. UPDATE

        # --- check the validation set (no learning happens here) ---
        model.eval()                           # dropout OFF
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()
        best_val = min(best_val, val_loss)     # remember the lowest

        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:3d} | "
                  f"train {loss.item():.3f} | val {val_loss:.3f}")
    return best_val

# -------------------------------------------------------------------
# STEP 4: Train both and compare
# -------------------------------------------------------------------
print("PLAIN model (no regularization):")
train(model_plain, opt_plain)

print("\nREGULARIZED model (dropout + weight decay):")
train(model_reg, opt_reg)

# -------------------------------------------------------------------
# STEP 5: Final, honest score on the locked-away TEST set
# -------------------------------------------------------------------
for name, model in [("plain", model_plain), ("regularized", model_reg)]:
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test), Y_test).item()
    print(f"{name:>12} model  ->  test loss: {test_loss:.3f}")
