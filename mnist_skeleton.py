import torch                                        ##数据
from torch import nn                                ##从 PyTorch 导入神经网络模块
from torch.utils.data import DataLoader             ##导入数据加载器
from torchvision import datasets,transforms         ##

DATE_ROOT = "/home/missxie/date/mnist"              ##数据集路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ##设置设备为 GPU（如果可用）或 CPU

## --数据--
transform=transforms.ToTensor()  ##将图像数据转换为张量
train_ds=datasets.MNIST(DATE_ROOT ,train=True,download=True,transform=transform)  ##加载训练数据集
test_ds=datasets.MNIST(DATE_ROOT ,train=False,download=True,transform=transform)  ##加载测试数据集

train_loader=DataLoader(train_ds,batch_size=64,shuffle=True)  ##创建训练数据加载器，批量大小为64，数据顺序打乱
test_loader=DataLoader(test_ds,batch_size=256,shuffle=False)  ##创建测试数据加载器，批量大小为256，数据顺序不打乱

# --模型--
model=nn.Sequential(
    nn.Flatten(),  ##将输入张量展平为一维
    nn.Linear(784, 128),  ##全连接层，将784维输入映射到128维
    nn.ReLU(),  ##ReLU激活函数
    nn.Linear(128, 10)  ## 输出 10 个分数，对应 0~9
).to(device)  ##使用顺序容器定义模型

print(model)  ##打印模型结构
print("参量数",sum(p.numel() for p in model.parameters()))  ##打印模型参数总数

##  --损失函数--
loss_fn=nn.CrossEntropyLoss()  ##定义交叉熵损失函数

##  --优化器--
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)  ##定义 Adam 优化器，学习率为 0.001

##  --训练循环--
for epoch in range(3):  ##训练3个周期
    ##  --训练--
    model.train()  ##设置模型为训练模式
    total_loss, seen = 0.0, 0  ##初始化总损失和已处理样本数
    for X,y in train_loader:  ##遍历训练数据加载器 
        X,y=X.to(device),y.to(device)  ##将输入数据和标签移动到设备上,搬上 GPU（框架价值②）
        y_pred=model(X)  ##前向传播，计算预测值
        loss=loss_fn(y_pred,y)  ##计算损失
        optimizer.zero_grad()  ##清零梯度
        loss.backward()  ##反向传播，计算梯度
        optimizer.step()  ##更新参数

        total_loss += loss.item() * y.size(0)  # .item() 把张量变普通数字
        seen += y.size(0)  # 统计样本数

    #--验证--
    model.eval()  ##设置模型为评估模式
    correct, total = 0, 0  ##初始化正确预测数和总样本数
    with torch.no_grad():  ##在验证阶段不需要计算梯度,省内存更快
        for X, y in test_loader:  ##遍历测试数据加载器
            X, y = X.to(device), y.to(device)  ##将输入数据和标签移动到设备上
            pred = model(X).argmax(dim=1)  ##前向传播，计算预测值
            correct += (pred == y).sum().item()  ##计算正确预测数
            total += y.size(0)  ##统计总样本数

    print(f"epoch {epoch+1}: 训练 Loss{total_loss/seen:.4f} | 验证周期 Accuracy{correct/total:.4f}")  ##打印每个周期的平均损失和准确率