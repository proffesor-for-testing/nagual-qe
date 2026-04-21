#!/usr/bin/env python3
"""
FastGRNN Router Model Training Script

This script trains a FastGRNN model for query complexity estimation.
The model is used to route queries to appropriate LLM vendors.

Usage:
    python train_fastgrnn.py --data trajectory_data.json --output fastgrnn_router.json
    python train_fastgrnn.py --export-onnx fastgrnn_router.onnx

Features (input to model):
    1. query_length: Normalized length of query (0-1)
    2. embedding_norm: L2 norm of query embedding (0-1)
    3. domain_specificity: Technical domain specificity score (0-1)
    4. pattern_coverage: How well patterns cover the query (0-1)
    5. historical_accuracy: Past accuracy on similar queries (0-1)

Output:
    complexity score [0.0, 1.0] for vendor routing:
    - < 0.3: local-small model
    - 0.3-0.5: local-large model
    - >= 0.5: cloud API (Claude/GPT)
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import random


@dataclass
class FastGRNNConfig:
    """Configuration for FastGRNN model."""
    input_dim: int = 5
    hidden_dim: int = 16
    output_dim: int = 1
    zeta: float = 1.0
    nu: float = -0.001
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


@dataclass
class FastGRNNWeights:
    """Weights for FastGRNN model."""
    w_z: List[float]  # Gate weights for input (hidden_dim x input_dim)
    u_z: List[float]  # Gate weights for hidden (hidden_dim x hidden_dim)
    b_z: List[float]  # Gate bias (hidden_dim)
    w_h: List[float]  # Hidden weights for input (hidden_dim x input_dim)
    u_h: List[float]  # Hidden weights for hidden (hidden_dim x hidden_dim)
    b_h: List[float]  # Hidden bias (hidden_dim)
    w_o: List[float]  # Output weights (output_dim x hidden_dim)
    b_o: List[float]  # Output bias (output_dim)
    zeta: float
    nu: float


class FastGRNN:
    """FastGRNN model for complexity estimation."""

    def __init__(self, config: FastGRNNConfig):
        self.config = config
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        c = self.config

        xavier_input = np.sqrt(6.0 / (c.input_dim + c.hidden_dim))
        xavier_hidden = np.sqrt(6.0 / (c.hidden_dim * 2))
        xavier_output = np.sqrt(6.0 / (c.hidden_dim + c.output_dim))

        self.W_z = np.random.uniform(-xavier_input, xavier_input, (c.hidden_dim, c.input_dim))
        self.U_z = np.random.uniform(-xavier_hidden, xavier_hidden, (c.hidden_dim, c.hidden_dim))
        self.b_z = np.zeros(c.hidden_dim)

        self.W_h = np.random.uniform(-xavier_input, xavier_input, (c.hidden_dim, c.input_dim))
        self.U_h = np.random.uniform(-xavier_hidden, xavier_hidden, (c.hidden_dim, c.hidden_dim))
        self.b_h = np.zeros(c.hidden_dim)

        self.W_o = np.random.uniform(-xavier_output, xavier_output, (c.output_dim, c.hidden_dim))
        self.b_o = np.zeros(c.output_dim)

        self.zeta = c.zeta
        self.nu = c.nu

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray, h_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim) or (input_dim,)
            h_prev: Previous hidden state (optional)

        Returns:
            (output, hidden_state)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]

        if h_prev is None:
            h_prev = np.zeros((batch_size, self.config.hidden_dim))

        # z_t = sigmoid(W_z * x + U_z * h_prev + b_z)
        z = self.sigmoid(x @ self.W_z.T + h_prev @ self.U_z.T + self.b_z)

        # h_tilde = tanh(W_h * x + U_h * h_prev + b_h)
        h_tilde = np.tanh(x @ self.W_h.T + h_prev @ self.U_h.T + self.b_h)

        # h_t = (zeta * (1 - z) + nu) * h_tilde + z * h_prev
        h = (self.zeta * (1 - z) + self.nu) * h_tilde + z * h_prev

        # output = sigmoid(W_o * h + b_o)
        output = self.sigmoid(h @ self.W_o.T + self.b_o)

        return output, h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict complexity score."""
        output, _ = self.forward(x)
        return output.flatten()

    def get_weights(self) -> FastGRNNWeights:
        """Export weights for Rust model."""
        return FastGRNNWeights(
            w_z=self.W_z.flatten().tolist(),
            u_z=self.U_z.flatten().tolist(),
            b_z=self.b_z.tolist(),
            w_h=self.W_h.flatten().tolist(),
            u_h=self.U_h.flatten().tolist(),
            b_h=self.b_h.tolist(),
            w_o=self.W_o.flatten().tolist(),
            b_o=self.b_o.tolist(),
            zeta=float(self.zeta),
            nu=float(self.nu)
        )

    def set_weights(self, weights: FastGRNNWeights):
        """Load weights from exported format."""
        c = self.config
        self.W_z = np.array(weights.w_z).reshape(c.hidden_dim, c.input_dim)
        self.U_z = np.array(weights.u_z).reshape(c.hidden_dim, c.hidden_dim)
        self.b_z = np.array(weights.b_z)
        self.W_h = np.array(weights.w_h).reshape(c.hidden_dim, c.input_dim)
        self.U_h = np.array(weights.u_h).reshape(c.hidden_dim, c.hidden_dim)
        self.b_h = np.array(weights.b_h)
        self.W_o = np.array(weights.w_o).reshape(c.output_dim, c.hidden_dim)
        self.b_o = np.array(weights.b_o)
        self.zeta = weights.zeta
        self.nu = weights.nu


def generate_synthetic_data(n_samples: int = 1000) -> List[Tuple[List[float], float]]:
    """
    Generate synthetic training data.

    Features:
        0. query_length: normalized length
        1. embedding_norm: semantic density
        2. domain_specificity: technical vs general
        3. pattern_coverage: existing pattern coverage
        4. historical_accuracy: past accuracy

    Label:
        complexity score based on feature combination
    """
    data = []

    for _ in range(n_samples):
        # Generate features
        query_length = random.random()
        embedding_norm = random.random()
        domain_specificity = random.random()
        pattern_coverage = random.random()
        historical_accuracy = random.random()

        features = [
            query_length,
            embedding_norm,
            domain_specificity,
            pattern_coverage,
            historical_accuracy
        ]

        # Compute complexity label based on features
        # Higher complexity when:
        # - Longer queries
        # - Higher embedding norm (more semantic content)
        # - Higher domain specificity
        # - Lower pattern coverage (fewer existing solutions)
        # - Lower historical accuracy (harder queries)

        complexity = (
            0.15 * query_length +
            0.15 * embedding_norm +
            0.30 * domain_specificity +
            0.20 * (1 - pattern_coverage) +
            0.20 * (1 - historical_accuracy)
        )

        # Add some noise
        complexity += random.gauss(0, 0.05)
        complexity = max(0.0, min(1.0, complexity))

        data.append((features, complexity))

    return data


def train_model(
    model: FastGRNN,
    data: List[Tuple[List[float], float]],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    verbose: bool = True
) -> List[float]:
    """
    Train the FastGRNN model using gradient descent.

    Returns:
        List of loss values per epoch
    """
    losses = []
    n_samples = len(data)

    for epoch in range(epochs):
        random.shuffle(data)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            X = np.array([d[0] for d in batch])
            y = np.array([d[1] for d in batch])

            # Forward pass
            pred, h = model.forward(X)
            pred = pred.flatten()

            # Compute loss (MSE)
            loss = np.mean((pred - y) ** 2)
            epoch_loss += loss * len(batch)

            # Backward pass (simplified gradient descent)
            # Gradient of MSE: 2 * (pred - y) / n
            grad_output = 2 * (pred - y) / len(batch)

            # Gradient for output layer
            # Average gradient across batch
            h_mean = h.mean(axis=0)  # (hidden_dim,)
            grad_output_mean = grad_output.mean()  # scalar
            grad_W_o = grad_output_mean * h_mean.reshape(1, -1)  # (1, hidden_dim)
            grad_b_o = grad_output_mean

            # Update output weights
            model.W_o -= learning_rate * grad_W_o
            model.b_o -= learning_rate * np.array([grad_b_o])

            # Simplified: Also update hidden layer weights slightly
            # (Full backprop through RNN is more complex)
            grad_h = grad_output_mean * model.W_o  # (1, hidden_dim)
            X_mean = X.mean(axis=0)  # (input_dim,)
            grad_W_h = grad_h.T @ X_mean.reshape(1, -1)  # (hidden_dim, input_dim)
            grad_b_h = grad_h.flatten()  # (hidden_dim,)

            model.W_h -= learning_rate * 0.1 * grad_W_h
            model.b_h -= learning_rate * 0.1 * grad_b_h

        epoch_loss /= n_samples
        losses.append(epoch_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

    return losses


def evaluate_model(model: FastGRNN, data: List[Tuple[List[float], float]]) -> dict:
    """Evaluate model performance."""
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    pred = model.predict(X)

    mse = np.mean((pred - y) ** 2)
    mae = np.mean(np.abs(pred - y))

    # Routing accuracy (correct vendor selection)
    def get_vendor(score):
        if score < 0.3:
            return 0  # local-small
        elif score < 0.5:
            return 1  # local-large
        else:
            return 2  # cloud

    pred_vendors = [get_vendor(p) for p in pred]
    true_vendors = [get_vendor(t) for t in y]
    routing_accuracy = np.mean([p == t for p, t in zip(pred_vendors, true_vendors)])

    return {
        "mse": float(mse),
        "mae": float(mae),
        "routing_accuracy": float(routing_accuracy)
    }


def export_to_onnx(model: FastGRNN, output_path: str, opset_version: int = 13):
    """
    Export FastGRNN model to ONNX format.

    This creates a simplified feedforward version of the model that computes
    the complexity score in a single forward pass (no recurrent state).

    Args:
        model: Trained FastGRNN model
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 13)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Error: PyTorch is required for ONNX export.")
        print("Install with: pip install torch")
        return False

    class FastGRNNTorch(nn.Module):
        """PyTorch version of FastGRNN for ONNX export."""

        def __init__(self, numpy_model: FastGRNN):
            super().__init__()
            c = numpy_model.config

            # Gate weights
            self.W_z = nn.Parameter(torch.from_numpy(numpy_model.W_z.astype(np.float32)))
            self.U_z = nn.Parameter(torch.from_numpy(numpy_model.U_z.astype(np.float32)))
            self.b_z = nn.Parameter(torch.from_numpy(numpy_model.b_z.astype(np.float32)))

            # Hidden weights
            self.W_h = nn.Parameter(torch.from_numpy(numpy_model.W_h.astype(np.float32)))
            self.U_h = nn.Parameter(torch.from_numpy(numpy_model.U_h.astype(np.float32)))
            self.b_h = nn.Parameter(torch.from_numpy(numpy_model.b_h.astype(np.float32)))

            # Output weights
            self.W_o = nn.Parameter(torch.from_numpy(numpy_model.W_o.astype(np.float32)))
            self.b_o = nn.Parameter(torch.from_numpy(numpy_model.b_o.astype(np.float32)))

            # Scalars
            self.zeta = numpy_model.zeta
            self.nu = numpy_model.nu
            self.hidden_dim = c.hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for single time step.

            Args:
                x: Input tensor of shape (batch_size, input_dim)

            Returns:
                Output tensor of shape (batch_size, 1)
            """
            batch_size = x.shape[0]

            # Initialize hidden state as zeros
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

            # z = sigmoid(W_z @ x + U_z @ h_prev + b_z)
            z = torch.sigmoid(
                torch.mm(x, self.W_z.T) +
                torch.mm(h_prev, self.U_z.T) +
                self.b_z
            )

            # h_tilde = tanh(W_h @ x + U_h @ h_prev + b_h)
            h_tilde = torch.tanh(
                torch.mm(x, self.W_h.T) +
                torch.mm(h_prev, self.U_h.T) +
                self.b_h
            )

            # h = (zeta * (1 - z) + nu) * h_tilde + z * h_prev
            h = (self.zeta * (1 - z) + self.nu) * h_tilde + z * h_prev

            # output = sigmoid(W_o @ h + b_o)
            output = torch.sigmoid(torch.mm(h, self.W_o.T) + self.b_o)

            return output

    # Create PyTorch model
    torch_model = FastGRNNTorch(model)
    torch_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, model.config.input_dim)

    # Export to ONNX
    try:
        torch.onnx.export(
            torch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Successfully exported ONNX model to: {output_path}")

        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model validation passed!")

            # Print model info
            print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
            print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

        except ImportError:
            print("Note: Install 'onnx' package to validate the exported model")
        except Exception as e:
            print(f"Warning: ONNX validation failed: {e}")

        return True

    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return False


def load_weights_from_json(json_path: str, config: FastGRNNConfig) -> FastGRNN:
    """Load a model from JSON weights file."""
    with open(json_path) as f:
        data = json.load(f)

    model = FastGRNN(config)
    weights = FastGRNNWeights(**data['weights'])
    model.set_weights(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train FastGRNN router model")
    parser.add_argument("--data", type=str, help="Path to training data JSON")
    parser.add_argument("--output", type=str, default="fastgrnn_router.json",
                        help="Output path for trained weights (JSON)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--samples", type=int, default=5000, help="Synthetic samples")
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="Export trained model to ONNX format")
    parser.add_argument("--load-json", type=str, default=None,
                        help="Load existing JSON weights for ONNX export (skip training)")
    args = parser.parse_args()

    # Configuration
    config = FastGRNNConfig(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        learning_rate=args.lr,
        epochs=args.epochs
    )

    print(f"FastGRNN Router Training")
    print(f"========================")
    print(f"Config: {asdict(config)}")
    print()

    # If loading existing weights for ONNX export only
    if args.load_json:
        print(f"Loading existing model from {args.load_json}...")
        model = load_weights_from_json(args.load_json, config)

        if args.export_onnx:
            export_to_onnx(model, args.export_onnx)
        else:
            print("No --export-onnx specified. Use --export-onnx <path> to export.")
        return

    # Load or generate data
    if args.data:
        print(f"Loading data from {args.data}...")
        with open(args.data) as f:
            raw_data = json.load(f)
        # Convert to training format
        data = [(d["features"], d["complexity"]) for d in raw_data]
    else:
        print(f"Generating {args.samples} synthetic samples...")
        data = generate_synthetic_data(args.samples)

    # Split data
    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    test_data = data[n_train:]

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print()

    # Initialize model
    model = FastGRNN(config)

    # Train
    print("Training...")
    losses = train_model(
        model, train_data,
        epochs=args.epochs,
        learning_rate=args.lr,
        verbose=True
    )
    print()

    # Evaluate
    print("Evaluation:")
    train_metrics = evaluate_model(model, train_data)
    test_metrics = evaluate_model(model, test_data)

    print(f"  Train MSE: {train_metrics['mse']:.6f}")
    print(f"  Train MAE: {train_metrics['mae']:.6f}")
    print(f"  Train Routing Accuracy: {train_metrics['routing_accuracy']:.2%}")
    print(f"  Test MSE: {test_metrics['mse']:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")
    print(f"  Test Routing Accuracy: {test_metrics['routing_accuracy']:.2%}")
    print()

    # Export weights to JSON
    weights = model.get_weights()
    output_data = {
        "config": asdict(config),
        "weights": asdict(weights),
        "metrics": {
            "train": train_metrics,
            "test": test_metrics
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Weights saved to {args.output}")

    # Model size
    n_params = (
        config.hidden_dim * config.input_dim * 2 +  # W_z, W_h
        config.hidden_dim * config.hidden_dim * 2 +  # U_z, U_h
        config.hidden_dim * 2 +  # b_z, b_h
        config.output_dim * config.hidden_dim +  # W_o
        config.output_dim +  # b_o
        2  # zeta, nu
    )
    size_bytes = n_params * 4  # float32
    print(f"Model size: {n_params} parameters, ~{size_bytes/1024:.1f} KB")

    # Export to ONNX if requested
    if args.export_onnx:
        print()
        print("Exporting to ONNX...")
        export_to_onnx(model, args.export_onnx)


if __name__ == "__main__":
    main()
