import numpy as np  #导入numpy库用于数值计算（做数组/矩阵运算的库）,起一个别名为np，就可以写np.xxxx,numpy 是 PyTorch 等一切的地基


rng = np.random.default_rng(0)     #创建一个随机数生成器对象 rng，使用默认的随机数生成器，种子为0，这样每次运行代码都会得到相同的随机数序列，便于调试和复现结果。
x= rng.uniform(0, 10, 100)         #从均匀分布中生成100个随机数，范围在0到10之间，作为输入特征 x
y = 2.0 * x + 1.0 + rng.normal(0, 1,size=x.shape)    #生成目标变量 y，y 的计算公式为 y = 2.0 * x + 1.0 + 噪声，其中噪声是从均值为0、标准差为1的正态分布中生成的随机数，大小与 x 相同。这样可以模拟一个线性关系，并加入一些随机噪声，使得数据更接近真实情况。

w, b = 0.0, 0.2
lr=0.01
epochs=1000

for epoch in range(epochs):    
    y_pred = w * x + b

    loss = ((y_pred - y)**2).mean()
    
    # Compute gradients
    dw = (2 * (y_pred - y) * x).mean()
    db = (2 * (y_pred - y)).mean()
    
    # Update weights and bias
    w -= lr * dw
    b -= lr * db
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}')