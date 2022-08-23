#%%

import torch

#%%

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"Using device {device}")

#%%

zeros = torch.zeros(1000,1,
                    requires_grad=False)
zeros.to(device)
for x in range(1000):
    zeros += zeros

#%%