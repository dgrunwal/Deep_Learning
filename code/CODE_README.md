# Deep Learning For Beginners — Code Reference

*Companion source files for* **Deep Learning For Beginners: A Practical Approach** *by David Grunwald.*

This folder contains every runnable program from the book, one `.py` file per program, named exactly as labelled in the book's script comments. Each file is self-contained: with PyTorch installed (`pip install torch numpy matplotlib`) you can run any of them directly, for example `python house_price.py`. The capstone, `mnist_digits.py`, also requires `torchvision` and downloads the MNIST dataset on first run.

All programs set `torch.manual_seed(42)` for reproducibility, so the numbers you see should closely match those printed in the book.

## File Index

| File | Chapter | Part | Description |
| --- | --- | --- | --- |
| `house_price.py` | 2 — Your First Neural Network | I | The simplest useful network. Trains a 4→8→1 model with one ReLU hidden layer to predict house prices from four features, then predicts the price of an unseen house. Introduces the five-step training loop (forward, loss, zero-grad, backward, step). |
| `house_price_v2.py` | 3 — Scaling Features and Testing Honestly | I | The Chapter 2 model upgraded with two real-world habits: feature scaling (standardize using training statistics only) and a held-back test set, so the model is graded on houses it never saw during training. |
| `tensor_basics.py` | 4 — How a Network Actually Learns | II | A standalone tour of the tensor, the core data structure of deep learning. Builds a small tensor, inspects its shape/dims/dtype, then indexes, reshapes (flatten), and scales it — the same operations later used on MNIST images. |
| `single_neuron.py` | 5 — Inside a Single Neuron | II | The smallest possible network: one neuron (one weight, one bias) taught the rule "double it." Demonstrates Linear layer, activation, loss, optimizer, and training loop shrunk to fit on one screen. |
| `backward_pass.py` | 7 — Backward Propagation: Learning From Mistakes | II | Shows backpropagation in action on a tiny network. Walks through how the loss flows backward to produce gradients and how the optimizer uses them to nudge the weights toward better predictions. |
| `shallow_classifier.py` | 8 — A Network With One Hidden Layer | III | A shallow network (one hidden layer) trained on a two-class, XOR-like problem that a single straight line cannot separate — demonstrating why hidden layers matter. Uses the same classification setup as the MNIST recognizer. |
| `activations_demo.py` | 9 — Choosing Activation Functions | III | Compares activation functions (e.g. ReLU, Sigmoid, Tanh) and shows their effect on a small classifier, ending with a class prediction via `argmax`. Builds intuition for which activation to reach for and why. |
| `deep_network.py` | 10 — From Shallow to Deep | IV | Builds a deep network with three hidden layers. Without training it, counts the network's parameters and pushes one example through to confirm data flows cleanly from input to output. Deliberately mirrors the shape of the MNIST recognizer. |
| `stable_training.py` | 11 — Keeping Training Stable | IV | Demonstrates techniques that keep training stable and well-behaved, including mini-batch settings and input normalization, reporting average batch loss per epoch. |
| `overfit_demo.py` | 12 — Overfitting and the Art of Generalization | IV | Makes overfitting visible by training two networks on the same small dataset: one plain (which overfits) and one regularized (dropout + weight decay). Compares their test loss to show regularization at work. |
| `image_classifier.py` | 13 — Working With Image Data | V | A tiny image classifier on hand-built 8×8 grayscale "images" (bright vs. dark), with no download required. Performs in miniature the full MNIST preparation pipeline: tensor conversion, pixel normalization, flattening, batching with a DataLoader, and Cross-Entropy training. |
| `mnist_digits.py` | 14 — Project: Recognizing Handwritten Digits | V | The capstone. Builds, trains, and evaluates a complete deep network on the full MNIST dataset (70,000 handwritten digit images), tests on held-out images, reports accuracy, and inspects the mistakes the network makes. Brings together tensors, layers, activations, the training loop, and honest evaluation. |

## Notes

- **Chapters 1, 6, and 15** contain no standalone program. Chapter 1 covers environment setup, Chapter 6 (Forward Propagation) is expository and shares machinery with the neuron and backward-pass examples, and Chapter 15 points to where to go next.
- **Expository vs. capstone.** The early files (`house_price.py`, `house_price_v2.py`) and the small "expository" programs build intuition piece by piece; everything converges on `mnist_digits.py`, the capstone project in Chapter 14.
- **Repository:** https://github.com/dgrunwal/Deep_Learning.git

© 2026 David Grunwald. All rights reserved.
