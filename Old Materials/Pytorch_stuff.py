#%%
import torch

#%%
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"Using device {device}")

#%%

n = int(1e6)
t = torch.randn(n, 100, requires_grad=False)
t = t * 1e-4
t = t.to(device)
print(f"Is tensor on GPU: {t.is_cuda}")

#%%

for n in range(10000):
    t = t + t
print(f"Is tensor on GPU: {t.is_cuda}")

#%%
