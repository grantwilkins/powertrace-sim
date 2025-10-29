"""Quick test to verify GRU classifier compatibility with dataset."""
import torch
from classifiers.gru import GRUClassifier

# Test with actual dimensions from dataset
# Dx=6 (from make_schedule_matrix: cnt, tok_in, tok_out, active_requests, prefill_tokens, decode_tokens)
# K=6 (number of power states)
# Batch size = 1, sequence length varies

print("Testing GRUClassifier compatibility...")

# Create model with default parameters
Dx = 6
K = 6
H = 64

model = GRUClassifier(Dx=Dx, K=K, H=H)
print(f"✓ Model created: Dx={Dx}, K={K}, H={H}")

# Test with sample input (batch=1, time=100, features=6)
x = torch.randn(1, 100, Dx)
model.eval()
with torch.no_grad():
    output = model(x)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {output.shape}")
assert output.shape == (1, 100, K), f"Expected (1, 100, {K}), got {output.shape}"
print(f"✓ Output shape correct: (B=1, T=100, K={K})")

# Test with different sequence lengths
for T in [10, 50, 200]:
    x = torch.randn(1, T, Dx)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, T, K)
    print(f"✓ Variable length T={T} works")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✓ Total parameters: {total_params:,}")
print(f"✓ Trainable parameters: {trainable_params:,}")

# Test training mode
model.train()
x = torch.randn(2, 50, Dx)  # batch=2
output = model(x)
assert output.shape == (2, 50, K)
print(f"✓ Training mode works with batch=2")

print("\n✅ All compatibility tests passed!")
print("The model is ready to use with your dataset.")
