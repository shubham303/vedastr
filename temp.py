import torch
import einops
y= torch.randn(3,2,4)

x = torch.randn(1,1,4)

x = einops.repeat(x, '() n e -> b n e', b=3 )
print(x.shape)
print(y.shape)
x = torch.cat([x , y],dim=1)

print(x)
