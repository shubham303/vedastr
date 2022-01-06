import torch

x = torch.ones([4, 2, 3, 6])
x[1,1,0,0]=3
x[1,1,1,1]=5
print(x.shape)
y = x.view(4,2,-1)
print(y.shape)

z= x.permute(0,1,3,2)
print(z.shape)
print(z.reshape(4,2,-1))

