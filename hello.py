import torch
from icecream import ic
from torch import nn

# x = torch.randn(3, 2)
# ic(x)
# y = x.unsqueeze(2)
# ic(y)
# ic(y.shape)

# ic(__name__)
# a = torch.randn(16, 7, 7, 1)
# ic(a[0])
# p = torch.load("predictions.pt")
# ic(p.shape)
# t = torch.load("targets.pt")
# ic(t.shape)
a = torch.randint(0, 3, size=(2, 4)).float()
ic(a)
b = torch.randint(0, 3, size=(2, 4)).float()
ic(b)
# b, c = torch.max(a, dim=0)
# ic(b, c)
# b = a[..., 1]
# ic(b.shape)
# ic(b)
# c = a * b
# ic(c.shape, c)
# b = torch.sign(a)
# ic(b)
mse = nn.MSELoss(reduction="sum")
c = mse(a, b)
ic(c)
# c = torch.flatten(a, end_dim=0)
# d = torch.flatten(a, end_dim=2)
# f = torch.flatten(a, end_dim=3)
# ic(c.shape, d.shape, f.shape)
