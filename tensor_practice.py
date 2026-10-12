import torch

a= torch.ones(2,3)

print((a + torch.ones(1, 3)).shape)
print((a + torch.arange(3.)).shape)
print((a * torch.ones(3, 1)).shape)
print((a @ torch.ones(3, 5)).shape)
