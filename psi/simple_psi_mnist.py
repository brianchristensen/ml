"""
Simplified PSI MNIST Classifier

Core idea: Biologically plausible learning using contrastive Hebbian learning
WITHOUT complex phase dynamics. Focus on what's essential:

1. Input projection: 784 pixels -> hidden layer
2. Hidden layer: Recurrent processing with lateral inhibition
3. Output: Hidden -> 10 classes
4. Learning: Contrastive Hebbian (free vs clamped phase)

This is a minimal architecture to verify the learning rule works before
adding back phase dynamics.
"""

import numpy as np
import time
from typing import Tuple


def load_mnist_subset(n_train: int = 1000, n_test: int = 500) -> Tuple:
    """Load MNIST data."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)

        # Normalize to [0, 1]
        X = X / 255.0

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        idx = np.random.permutation(len(X_train))[:n_train]
        X_train, y_train = X_train[idx], y_train[idx]

        if n_test < len(X_test):
            idx = np.random.permutation(len(X_test))[:n_test]
            X_test, y_test = X_test[idx], y_test[idx]

        print(f"Loaded {len(X_train)} train, {len(X_test)} test samples")
        return X_train, y_train, X_test, y_test
    except ImportError:
        np.random.seed(42)
        X_train = np.random.randn(n_train, 784) * 0.3
        y_train = np.random.randint(0, 10, n_train)
        X_test = np.random.randn(n_test, 784) * 0.3
        y_test = np.random.randint(0, 10, n_test)
        return X_train, y_train, X_test, y_test


class SimplePSIClassifier:
    """
    Simple contrastive Hebbian classifier.

    Architecture:
    - Input: 784 pixels (raw MNIST)
    - Hidden: n_hidden units with lateral inhibition
    - Output: 10 units (one per class)

    Learning: Contrastive Hebbian
    - Free phase: Input clamped, network settles
    - Target phase: Input + output clamped to target
    - Learning: Strengthen connections active in target phase,
                weaken those active only in free phase
    """

    def __init__(self, n_hidden: int = 100, n_output: int = 10, lr: float = 0.01, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_input = 784
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr

        # Weights (Xavier initialization)
        self.W_ih = np.random.randn(self.n_input, n_hidden) * np.sqrt(2.0 / self.n_input)
        self.W_ho = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)

        # Biases
        self.b_h = np.zeros(n_hidden)
        self.b_o = np.zeros(n_output)

        # Lateral inhibition in hidden layer (sparse, winner-take-all style)
        self.W_hh = -0.1 * np.ones((n_hidden, n_hidden))
        np.fill_diagonal(self.W_hh, 0)  # No self-connection

        # Feedback alignment: random fixed feedback weights (biologically plausible alternative to backprop)
        self.B = np.random.randn(n_output, n_hidden) * 0.1  # Fixed, not learned

        # States
        self.h = np.zeros(n_hidden)  # Hidden activations
        self.o = np.zeros(n_output)  # Output activations

    def activate(self, x):
        """Activation function (ReLU)."""
        return np.maximum(0, x)

    def free_phase(self, x: np.ndarray, n_steps: int = 5) -> np.ndarray:
        """
        Free phase: Input clamped, let hidden and output settle.
        Returns output activations.
        """
        # Initialize hidden from input
        self.h = self.activate(x @ self.W_ih + self.b_h)

        # Settle with lateral inhibition
        for _ in range(n_steps):
            h_input = x @ self.W_ih + self.h @ self.W_hh + self.b_h
            self.h = 0.5 * self.h + 0.5 * self.activate(h_input)

        # Compute output
        self.o = self.h @ self.W_ho + self.b_o

        # Store free phase states
        self.h_free = self.h.copy()
        self.o_free = self.o.copy()

        return self.o

    def target_phase(self, x: np.ndarray, target: int, n_steps: int = 5):
        """
        Target phase: Input and output clamped to target.
        Hidden layer settles to match both constraints.
        """
        # Create target output (one-hot)
        target_o = np.zeros(self.n_output)
        target_o[target] = 1.0

        # Start from free phase hidden state
        self.h = self.h_free.copy()

        # Settle with both input and output clamped
        for _ in range(n_steps):
            # Input from below (from pixels)
            h_bottom = x @ self.W_ih + self.b_h
            # Input from above (from clamped output, transposed weights)
            h_top = target_o @ self.W_ho.T
            # Lateral inhibition
            h_lateral = self.h @ self.W_hh

            h_total = h_bottom + 0.5 * h_top + h_lateral
            self.h = 0.5 * self.h + 0.5 * self.activate(h_total)

        # Store target phase states
        self.h_target = self.h.copy()
        self.o_target = target_o.copy()

    def softmax(self, x):
        """Softmax activation for output."""
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-8)

    def learn(self, x: np.ndarray, label: int):
        """
        Hybrid learning with feedback alignment:
        - W_ho: Direct error-driven (like delta rule)
        - W_ih: Feedback alignment (error propagated through random fixed weights B)
        """
        # Output error (softmax gradient)
        probs = self.softmax(self.o_free)
        o_error = probs.copy()
        o_error[label] -= 1.0

        # W_ho: Direct gradient descent (error-driven)
        dW_ho = -self.lr * np.outer(self.h_free, o_error)
        self.W_ho += dW_ho
        self.b_o -= self.lr * o_error

        # W_ih: Feedback alignment (biologically plausible)
        # Error is propagated through random fixed weights B instead of W_ho.T
        h_error = o_error @ self.B  # Random projection of output error
        h_error[self.h_free <= 0] = 0  # ReLU derivative

        dW_ih = -self.lr * np.outer(x, h_error)
        self.W_ih += dW_ih
        self.b_h -= self.lr * h_error

        # Weight decay for regularization
        self.W_ih *= 0.999
        self.W_ho *= 0.999

    def train_step(self, x: np.ndarray, label: int):
        """One training step."""
        self.free_phase(x)
        self.target_phase(x, label)
        self.learn(x, label)

    def predict(self, x: np.ndarray) -> int:
        """Predict class for input."""
        outputs = self.free_phase(x)
        return int(np.argmax(outputs))


class MLPBaseline:
    """Simple MLP for comparison."""

    def __init__(self, hidden_size: int = 100, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.W1 = np.random.randn(784, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 10) * 0.01
        self.b2 = np.zeros(10)
        self.lr = 0.01

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        return h, o

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def train_step(self, x, label):
        h, o = self.forward(x)
        probs = self.softmax(o)

        do = probs.copy()
        do[label] -= 1

        dW2 = np.outer(h, do)
        db2 = do

        dh = do @ self.W2.T
        dh[h <= 0] = 0

        dW1 = np.outer(x, dh)
        db1 = dh

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, x):
        _, o = self.forward(x)
        return int(np.argmax(o))


def evaluate(model, X_test, y_test, name: str) -> float:
    correct = sum(1 for i in range(len(X_test)) if model.predict(X_test[i]) == y_test[i])
    acc = correct / len(X_test)
    print(f"{name}: {correct}/{len(X_test)} = {acc:.1%}")
    return acc


def main():
    print("=" * 70)
    print("SIMPLE PSI vs MLP BENCHMARK")
    print("=" * 70)
    print()

    np.random.seed(42)

    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=1000, n_test=500)

    print()
    print("Training Simple PSI classifier...")
    print("-" * 50)

    psi = SimplePSIClassifier(n_hidden=100, lr=0.01, seed=42)

    psi_start = time.time()
    for epoch in range(5):
        epoch_start = time.time()
        idx = np.random.permutation(len(X_train))

        for i in idx:
            psi.train_step(X_train[i], y_train[i])

        epoch_time = time.time() - epoch_start

        # Quick eval
        test_idx = np.random.permutation(len(X_test))[:100]
        correct = sum(1 for i in test_idx if psi.predict(X_test[i]) == y_test[i])
        print(f"  Epoch {epoch+1}: {correct}/100 = {correct}% ({epoch_time:.1f}s)")

    psi_time = time.time() - psi_start
    print(f"\nPSI training time: {psi_time:.1f}s")

    print()
    print("Training MLP baseline...")
    print("-" * 50)

    mlp = MLPBaseline(hidden_size=100, seed=42)

    mlp_start = time.time()
    for epoch in range(5):
        epoch_start = time.time()
        idx = np.random.permutation(len(X_train))
        for i in idx:
            mlp.train_step(X_train[i], y_train[i])

        epoch_time = time.time() - epoch_start

        test_idx = np.random.permutation(len(X_test))[:100]
        correct = sum(1 for i in test_idx if mlp.predict(X_test[i]) == y_test[i])
        print(f"  Epoch {epoch+1}: {correct}/100 = {correct}% ({epoch_time:.1f}s)")

    mlp_time = time.time() - mlp_start
    print(f"MLP training time: {mlp_time:.1f}s")

    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    psi_acc = evaluate(psi, X_test, y_test, "PSI")
    mlp_acc = evaluate(mlp, X_test, y_test, "MLP")

    print()
    print("Summary:")
    print(f"  PSI: {psi_acc:.1%} accuracy, {psi_time:.1f}s training")
    print(f"  MLP: {mlp_acc:.1%} accuracy, {mlp_time:.1f}s training")

    if psi_acc >= mlp_acc:
        print("\n  PSI matches or beats MLP!")
    else:
        print(f"\n  MLP wins by {(mlp_acc - psi_acc)*100:.1f} percentage points")


if __name__ == "__main__":
    main()
