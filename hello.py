import torch
from icecream import ic
from torch import nn
from collections import Counter

# for i in range(10):
#     ic(i)
# a = torch.randint(0, 10, size=(3, 3, 3))
# ic(a)
# b = torch.argmax(a)
# c = torch.argmax(a, dim=-1)
# d = torch.argmax(a, dim=-1, keepdim=True)
# f = torch.argmax(a, dim=-1).unsqueeze(-1)
# ic(b, c, d, f)
# ic(c.shape, d.shape, f.shape)

# a = torch.randint(0, 9, size=(3, 3, 3, 3))
# ic(a)
# b = torch.randint(-9, 0, size=(3, 3))
# ic(a, b)
# c = torch.cat([a, b])
# ic(c, c.shape)
# d = torch.cat([a, b], dim=-1)
# ic(d, d.shape)
# b = a.repeat(1, 1, 1, 4)
# ic(b, b.shape)
# b = torch.arange(7)
# ic(b, b.shape)
# c = b.repeat(16, 7, 1)
# ic(c, c.shape)
# d = c.unsqueeze(-1)
# ic(d, d.shape)
# f = d.transpose(1, 2)
# ic(f, f.shape)
# a = torch.randint(0, 9, size=(2, 3, 5))
# ic(a)
# b = a.transpose(1, 2)
# ic(b)
# ic(a.shape, b.shape)

# a = [1, 2, 3, 4, 5]
# b = torch.tensor(a[1:])
# ic(b.shape)

# ground_truths = torch.randint()
# c1 = Counter("banana")
# ic(c1)
# a = [1, 2, 3, 1, 2, 5, 1, 2]
# b = Counter(a)
# ic(b)

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
ic(a.shape)

print(torch.cumsum(a, dim=0))
