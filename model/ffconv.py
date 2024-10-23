import torch
import torch.nn as nn
from dataset.chlora import ChloraData

class FlourierConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FlourierConv3d, self).__init__()
        
        # 定义局部分支
        self.local_branch = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        
        # 定义半全局分支
        self.freq_conv = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=1)
        
        # 定义全局分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        # 计算局部分支输出
        local_out = self.local_branch(x)
        
        # 计算半全局分支输出
        freq_feature_map = torch.fft.fftn(local_out, dim=(-2,-1))
        freq_feature_map = self.freq_conv(freq_feature_map.real).unsqueeze(-1) + \
                           self.freq_conv(freq_feature_map.imag).unsqueeze(-1) * 1j
        semi_global_out = torch.fft.ifftn(freq_feature_map.real).squeeze(-1)
        
        # 计算全局分支输出
        global_pool_out = self.global_pool(local_out).view(local_out.size(0), -1)
        global_out = self.fc(global_pool_out).view(local_out.size(0), -1, 1, 1)
        
        # 将三个分支的输出进行融合并返回结果
        return torch.cat([local_out, semi_global_out, global_out], dim=1)


# 定义一个包含FFC层的神经网络模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.ffc1 = FlourierConv3d(64, 128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ffc2 = FlourierConv3d(256, 512)
        self.fc = nn.Linear(512*4*4, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.ffc1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.ffc2(x)
        x = x.view(-1, 512*4*4)
        x = self.fc(x)
        return x
if __name__=="__main__":
    # 创建一个MyModel实例并进行训练
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 1
    train_loader = ChloraData()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()