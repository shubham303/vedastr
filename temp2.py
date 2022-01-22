import torch

a = torch.ones(size=(4,  2))
b = torch.ones(size=(4,  3))
b[2]=3

print (a*b)