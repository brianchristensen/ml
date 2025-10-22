"""
Pure Holographic Associative Memory (HAM) Implementation

Based on:
1. Holographic Associative Memory - complex phase encoding
2. HDRAM/Hypertokens - phase-coherent retrieval
3. No learned parameters - episodic storage only

Architecture:
    Input -> Fixed Random Projection -> Complex Plane -> Holographic Storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HolographicMemory(nn.Module):
    """
    Pure HAM implementation using complex-valued storage.

    Storage: H = Stimulus*.T @ Response (conjugate transpose)
    Retrieval: Response = Query*.T @ H

    Information encoded in phase angles, confidence in magnitude.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Complex-valued holographic memory matrix
        # Using buffer (not parameter) - updated via .store(), NOT gradients
        self.register_buffer('memory', torch.zeros(dim, dim, dtype=torch.cfloat))

        # Track memory "fill level"
        self.register_buffer('energy', torch.tensor(0.0))
        self.register_buffer('num_stored', torch.tensor(0))

    def store(self, stimulus_complex, response_complex, learning_rate=0.1):
        """
        HAM storage via complex conjugate transpose.

        As per literature: Simple addition, no decay.

        Args:
            stimulus_complex: [batch, dim] complex tensor (unit norm)
            response_complex: [batch, dim] complex tensor (unit norm)
            learning_rate: How strongly to encode
        """
        # Average across batch first (superposition)
        stim_avg = stimulus_complex.mean(0)  # [dim]
        resp_avg = response_complex.mean(0)  # [dim]

        # HAM storage: H += alpha * (S*.T @ R)
        # Outer product: [dim, 1] @ [1, dim] = [dim, dim]
        update = torch.outer(stim_avg.conj(), resp_avg)

        # Simple addition (superposition)
        self.memory = self.memory + learning_rate * update

        # Normalize memory matrix columnwise (as per Dense Associative Memory)
        # This keeps retrieval bounded
        col_norms = torch.norm(self.memory, dim=0, keepdim=True)
        self.memory = self.memory / (col_norms + 1e-8)

        # Track statistics
        self.energy.copy_(self.memory.abs().sum())
        self.num_stored += stimulus_complex.size(0)

    def retrieve(self, query_complex):
        """
        HAM retrieval via complex conjugate transpose.

        Args:
            query_complex: [batch, dim] complex tensor

        Returns:
            response: [batch, dim] complex tensor
            confidence: [batch] float tensor (phase coherence measure)
        """
        batch_size = query_complex.size(0)

        # Retrieval: R = Q*.T @ H
        # [batch, dim] @ [dim, dim] = [batch, dim]
        response = query_complex.conj() @ self.memory.T

        # Phase coherence as confidence measure
        # High coherence = strong match, low = weak/noisy
        query_norm = query_complex.abs().sum(dim=-1, keepdim=True) + 1e-8
        response_norm = response.abs().sum(dim=-1, keepdim=True) + 1e-8

        # Real part of normalized inner product = phase alignment
        coherence = (query_complex.conj() * response).real.sum(dim=-1)
        confidence = coherence / (query_norm.squeeze() * response_norm.squeeze())

        return response, confidence


class HolographicClassifier(nn.Module):
    """
    Classification via per-class holographic memories.

    Each class has its own holographic memory storing exemplars.
    Classification = find class with highest correlation.

    Zero learnable parameters!

    Implements proper HAM preprocessing:
    - Orthogonalization (whitening)
    - Gaussian normalization
    - Unit sphere projection in complex domain
    """
    def __init__(
        self,
        input_dim=784,
        memory_dim=4096,
        num_classes=10,
        projection_scale=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_classes = num_classes

        # FIXED random projection (high-dimensional expansion)
        # This is our "feature extractor" - never updated!
        self.register_buffer(
            'random_projection',
            torch.randn(input_dim, memory_dim) * projection_scale
        )


        # Per-class holographic memories
        # Note: HolographicMemory takes COMPLEX dimension = memory_dim // 2
        complex_dim = memory_dim // 2
        self.class_memories = nn.ModuleList([
            HolographicMemory(dim=complex_dim)
            for _ in range(num_classes)
        ])

    def to_complex(self, x):
        """
        Map real-valued vector to complex plane.

        Strategy: Split dimensions in half, use as real/imaginary parts.
        This preserves all information while enabling phase encoding.
        """
        # Ensure even dimension
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1))

        mid = x.size(-1) // 2
        real = x[..., :mid]
        imag = x[..., mid:]

        complex_vec = torch.complex(real, imag)

        # L2 normalize to unit norm (as per Dense Associative Memory literature)
        complex_norm = complex_vec / (torch.norm(complex_vec, dim=-1, keepdim=True) + 1e-8)

        return complex_norm

    def process_input(self, x):
        """
        Fixed transformation: Input -> High-dim -> Complex (with L2 norm)

        As per Dense Associative Memory literature:
        - L2 normalize inputs to unit norm (done in to_complex())
        - Memory columns normalized in store()

        Args:
            x: [batch, input_dim] real tensor

        Returns:
            complex_features: [batch, memory_dim//2] complex tensor
        """
        # Random projection (fixed!)
        h = torch.tanh(x @ self.random_projection)

        # Map to complex plane (includes L2 normalization to unit norm)
        complex_features = self.to_complex(h)

        return complex_features

    def store(self, x, labels):
        """
        Store patterns in appropriate class memories.

        Args:
            x: [batch, input_dim] input patterns
            labels: [batch] class labels
        """
        # Process input (fixed transformation)
        complex_features = self.process_input(x)

        # Store in each class memory (no gradients!)
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                # Get samples belonging to this class
                mask = (labels == class_idx)
                if mask.any():
                    class_features = complex_features[mask]

                    # Store in holographic memory
                    # Response = same as stimulus (auto-associative)
                    self.class_memories[class_idx].store(
                        class_features,
                        class_features,
                        learning_rate=0.1
                    )

    def forward(self, x, labels=None, learn=True):
        """
        Forward pass: store (if learning) and classify.

        Args:
            x: [batch, input_dim] input patterns
            labels: [batch] class labels (needed for training)
            learn: Whether to store patterns in memory

        Returns:
            dict with 'logits', 'confidence', 'memory_stats'
        """
        batch_size = x.size(0)

        # Process input
        complex_features = self.process_input(x)

        # Store in memories if learning
        if learn and labels is not None:
            self.store(x, labels)

        # Classification: correlate with all class memories
        all_responses = []
        all_confidences = []

        for class_idx in range(self.num_classes):
            response, confidence = self.class_memories[class_idx].retrieve(complex_features)
            all_responses.append(response)
            all_confidences.append(confidence)

        # Stack: [num_classes, batch, memory_dim//2]
        all_responses = torch.stack(all_responses)  # [num_classes, batch, dim]
        all_confidences = torch.stack(all_confidences)  # [num_classes, batch]

        # Correlation strength as logits
        # Higher correlation = higher confidence = higher logit
        # Use response magnitude as similarity measure
        response_magnitudes = all_responses.abs().mean(dim=-1)  # [num_classes, batch]

        # Transpose to [batch, num_classes] for standard format
        logits = response_magnitudes.T
        confidences = all_confidences.T

        # Scale logits for reasonable cross-entropy range
        logits = logits * 10.0

        # Memory statistics
        memory_energies = torch.tensor([
            mem.energy.item() for mem in self.class_memories
        ])

        return {
            'logits': logits,
            'confidences': confidences,
            'memory_energy_avg': memory_energies.mean().item(),
            'memory_energy_max': memory_energies.max().item(),
            'num_stored': sum(mem.num_stored.item() for mem in self.class_memories)
        }
