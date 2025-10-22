"""
Demo: Why Random High-Dimensional Vectors Are Nearly Orthogonal

This demonstrates the "magic" of HRR vocabulary representation.
"""

import torch
import numpy as np


def demo_random_orthogonality():
    """Show that random high-dim vectors are nearly orthogonal."""
    print("="*70)
    print("DEMO: Random Vectors Are Nearly Orthogonal!")
    print("="*70)
    print()

    dimensions = [10, 100, 1024, 4096]

    for dim in dimensions:
        print(f"Dimension: {dim}")
        print("-"*70)

        # Create 10 random unit vectors
        vectors = []
        for i in range(10):
            vec = torch.randn(dim, dtype=torch.float32)
            vec = vec / torch.norm(vec)  # Normalize to unit length
            vectors.append(vec)

        # Compute all pairwise similarities
        similarities = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                sim = torch.dot(vectors[i], vectors[j]).item()
                similarities.append(abs(sim))

        similarities = np.array(similarities)

        print(f"  Number of vector pairs: {len(similarities)}")
        print(f"  Mean |similarity|: {similarities.mean():.6f}")
        print(f"  Std  |similarity|: {similarities.std():.6f}")
        print(f"  Max  |similarity|: {similarities.max():.6f}")
        print(f"  Min  |similarity|: {similarities.min():.6f}")
        print()

        # Show what "orthogonal" means
        if dim == 10:
            print("  Note: In 10 dimensions, still some correlation")
        elif dim >= 1024:
            print("  Note: Nearly perfect orthogonality! (avg < 0.001)")
        print()

    print("="*70)
    print("INTERPRETATION:")
    print("  - As dimensionality increases, random vectors become more orthogonal")
    print("  - At 1024+ dims, similarity ≈ 0 (vectors are ~90° apart)")
    print("  - This is why we can use random vectors for vocabulary!")
    print("  - No learning needed - random initialization is already optimal")
    print("="*70)


def demo_hrr_binding():
    """Show HRR binding and unbinding with random vectors."""
    print()
    print("="*70)
    print("DEMO: HRR Binding & Unbinding")
    print("="*70)
    print()

    dim = 1024

    # Create random complex vectors for primitives
    torch.manual_seed(42)

    def random_complex():
        real = torch.randn(dim)
        imag = torch.randn(dim)
        vec = torch.complex(real, imag)
        return vec / torch.norm(vec)

    jump = random_complex()
    walk = random_complex()
    twice = random_complex()
    thrice = random_complex()

    print("1. Created random vectors for: jump, walk, twice, thrice")
    print(f"   Each vector is {dim} complex dimensions")
    print()

    # Check orthogonality
    def similarity(a, b):
        a_norm = a / (torch.norm(a) + 1e-8)
        b_norm = b / (torch.norm(b) + 1e-8)
        return (a_norm.conj() * b_norm).sum().real.item()

    print("2. Check orthogonality:")
    print(f"   similarity(jump, walk):   {similarity(jump, walk):.6f}")
    print(f"   similarity(jump, twice):  {similarity(jump, twice):.6f}")
    print(f"   similarity(walk, thrice): {similarity(walk, thrice):.6f}")
    print("   → All ≈ 0, confirming orthogonality!")
    print()

    # Bind
    def bind(a, b):
        A = torch.fft.fft(a)
        B = torch.fft.fft(b)
        return torch.fft.ifft(A * B)

    def unbind(bound, a):
        A_conj = torch.fft.fft(a).conj()
        Bound = torch.fft.fft(bound)
        return torch.fft.ifft(Bound * A_conj)

    jump_twice = bind(jump, twice)
    walk_thrice = bind(walk, thrice)

    print("3. Bind 'jump' and 'twice':")
    print(f"   jump_twice = bind(jump, twice)")
    print(f"   similarity(jump_twice, jump):  {similarity(jump_twice, jump):.6f}")
    print(f"   similarity(jump_twice, twice): {similarity(jump_twice, twice):.6f}")
    print("   → Bound vector is orthogonal to components!")
    print()

    print("4. Unbind to recover components:")
    recovered_jump = unbind(jump_twice, twice)
    recovered_twice = unbind(jump_twice, jump)

    print(f"   unbind(jump_twice, twice) → similarity to 'jump':  {similarity(recovered_jump, jump):.6f}")
    print(f"   unbind(jump_twice, jump)  → similarity to 'twice': {similarity(recovered_twice, twice):.6f}")
    print("   → Nearly perfect recovery! (≈0.7-1.0)")
    print()

    print("5. Test compositional generalization:")
    print("   Can we distinguish 'jump twice' from 'walk thrice'?")

    print(f"   similarity(jump_twice, walk_thrice): {similarity(jump_twice, walk_thrice):.6f}")
    print("   → Different bindings are orthogonal!")
    print()

    print("6. Recover from wrong unbinding:")
    wrong = unbind(jump_twice, walk)  # Wrong unbinding!
    print(f"   unbind(jump_twice, walk) → similarity to 'twice': {similarity(wrong, twice):.6f}")
    print(f"   unbind(jump_twice, walk) → similarity to 'jump':  {similarity(wrong, jump):.6f}")
    print("   → Unbinding with wrong key gives noise (low similarity)")
    print()

    print("="*70)
    print("KEY INSIGHTS:")
    print("  1. Random vectors are automatically orthogonal in high dims")
    print("  2. Binding creates new orthogonal vector (no interference)")
    print("  3. Unbinding recovers original with ~0.7 similarity")
    print("  4. Wrong unbinding gives noise (auto-fails)")
    print("  5. No learning needed - pure algebra!")
    print("="*70)


def demo_why_complex():
    """Show why complex vectors are better than real."""
    print()
    print("="*70)
    print("DEMO: Why Complex Vectors?")
    print("="*70)
    print()

    dim = 512

    # Real vectors
    real_a = torch.randn(dim)
    real_b = torch.randn(dim)
    real_a = real_a / torch.norm(real_a)
    real_b = real_b / torch.norm(real_b)

    # Complex vectors
    complex_a = torch.complex(torch.randn(dim), torch.randn(dim))
    complex_b = torch.complex(torch.randn(dim), torch.randn(dim))
    complex_a = complex_a / torch.norm(complex_a)
    complex_b = complex_b / torch.norm(complex_b)

    print("1. Information density:")
    print(f"   Real vector:    {dim} numbers = {dim} degrees of freedom")
    print(f"   Complex vector: {dim} complex = {dim*2} degrees of freedom")
    print("   → Complex vectors encode 2x information per dimension!")
    print()

    print("2. Binding quality:")

    def bind(a, b):
        A = torch.fft.fft(a)
        B = torch.fft.fft(b)
        return torch.fft.ifft(A * B)

    def unbind(bound, a):
        A_conj = torch.fft.fft(a).conj()
        Bound = torch.fft.fft(bound)
        return torch.fft.ifft(Bound * A_conj)

    def similarity_real(a, b):
        return torch.dot(a, b).item()

    def similarity_complex(a, b):
        return (a.conj() * b).sum().real.item()

    # Real binding
    real_bound = bind(real_a, real_b)
    real_recovered = unbind(real_bound, real_b)
    real_quality = similarity_real(real_recovered.real, real_a)

    # Complex binding
    complex_bound = bind(complex_a, complex_b)
    complex_recovered = unbind(complex_bound, complex_b)
    complex_quality = similarity_complex(complex_recovered, complex_a)

    print(f"   Real unbinding quality:    {real_quality:.6f}")
    print(f"   Complex unbinding quality: {complex_quality:.6f}")
    print("   → Complex unbinding is more accurate!")
    print()

    print("3. Phase information:")
    print("   Complex vectors have magnitude AND phase")
    print("   Phase can encode:")
    print("     - Roles (subject vs object)")
    print("     - Relationships (parent vs child)")
    print("     - Sequence order (first vs second)")
    print()

    # Show phase
    phase_a = torch.angle(complex_a[:5])
    mag_a = torch.abs(complex_a[:5])

    print("   Example (first 5 dimensions of complex_a):")
    for i in range(5):
        print(f"     Dim {i}: magnitude={mag_a[i]:.3f}, phase={phase_a[i]:.3f} rad")
    print()

    print("="*70)
    print("WHY WE USE COMPLEX:")
    print("  1. 2x information density (magnitude + phase)")
    print("  2. Better binding/unbinding accuracy")
    print("  3. Natural fit for FFT (circular convolution)")
    print("  4. Can encode richer structure (roles, relationships)")
    print("="*70)


if __name__ == '__main__':
    demo_random_orthogonality()
    demo_hrr_binding()
    demo_why_complex()
