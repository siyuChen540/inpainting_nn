import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class DDPM(nn.Module):
    def __init__(self, in_channels, channel_width):
        super().__init__()
        self.in_channels = in_channels
        self.channel_width = channel_width
        
        self.conv1 = nn.Conv2d(in_channels, channel_width, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(channel_width) for _ in range(4)])
        self.conv2 = nn.Conv2d(channel_width, in_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, noise_level, reverse=False):
        if not reverse:
            noise = torch.randn_like(x) * noise_level
            x = x + noise
            log_det = 0.5 * x.numel() * math.log(2 * math.pi) + torch.sum(torch.log(noise_level) + (noise ** 2) / (2 * noise_level ** 2))
        else:
            log_det = None
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        return x, log_det

class ResBlock(nn.Module):
    def __init__(self, channel_width):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_width, channel_width, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_width, channel_width, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(channel_width)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.norm(h)
        h = self.leaky_relu(h)
        h = self.conv2(h)
        h = self.norm(h)
        return h + x


# Define the loss function
def loss_function(x, noise_level):
    y, log_det = model(x, noise_level)
    loss = F.mse_loss(y, x)
    loss = loss + log_det
    return loss

if __name__ == '__main__':
    # Load the data
    train_data = ... # Your training data
    test_data = ... # Your testing data

    # Define the model
    model = DDPM(in_channels=3, channel_width=64)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Train the model
    num_epochs = 100
    batch_size = 32
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            noise_level = 0.1 # You can set the noise level as per your requirement
            loss = loss_function(batch, noise_level)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Evaluate the model after each epoch
        with torch.no_grad():
            test_batch = test_data[:batch_size]
            y, _ = model(test_batch, noise_level)
            test_loss = F.mse_loss(y, test_batch)
            print(f"Epoch {epoch+1}, Test Loss: {test_loss.item()}")
