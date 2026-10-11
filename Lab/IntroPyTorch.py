import torch

a= torch.tensor([[1, 2], [3, 4]])
print(a)
print("shape",a.shape,"| dtype",a.dtype,"| device",a.device)  #shape = 长什么样（几行几列），dtype = 装什么数（整数/小数），device = 住在哪（CPU/GPU）。

print("a+1=",a+1)
print("a*1=",a*a)   
print("a@a=",a@a)

print("zeroes:",torch.zeros(2,3))                       #zeros表示里面装的全是0，ones全是1，randn随机小数(正态分布)，rand随机小数(均匀分布)
print("randn:",torch.randn(3,4))
print("广播:",torch.zeros(2,3)+ torch.randn(1,3))


device ='cuda'if torch.cuda.is_available() else 'cpu'  #判断是否有GPU可用，如果有就用GPU，否则用CPU
a=a.to(device)  #将张量a移动到指定的设备上（CPU或GPU），以便在该设备上进行计算
print("a device:",a.device)  #打印张量a所在的设备信息