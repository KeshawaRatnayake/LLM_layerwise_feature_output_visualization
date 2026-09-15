import torch
hs = torch.load("hidden_states.pt")
print(len(hs))

print(hs[6].shape)
