import torch
import torch.nn as nn

class FeatureNet(nn.Module):
    
    def __init__(self, in_channel, device):
        super(FeatureNet, self).__init__()
        
        # CNNs
        self.conv1 = nn.Conv2d(in_channel, 32, (3, 3)).to(device=device)
        self.conv2 = nn.Conv2d(in_channel, 32, (5, 5), padding=1).to(device=device)
        self.conv3 = nn.Conv2d(in_channel, 32, (7, 7), padding=2).to(device=device)
        self.conv4 = nn.Conv2d(in_channel, 32, (11, 11), padding=4).to(device=device)
        self.conv5 = nn.Conv2d(in_channel, 32, (15, 15), padding=6).to(device=device)
        self.conv6 = nn.Conv2d(in_channel, 32, (19, 19), padding=8).to(device=device)
        self.conv7 = nn.Conv2d(in_channel, 32, (23, 23), padding=10).to(device=device)
        self.conv8 = nn.Conv2d(in_channel, 32, (27, 27), padding=12).to(device=device)
        
        self.cnn_lst = [self.conv1, 
                        self.conv2,
                        self.conv3, 
                        self.conv4,
                        self.conv5, 
                        self.conv6,
                        self.conv7,
                        self.conv8]
        
        # ReLU
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        maps = []
        
        for cnn in self.cnn_lst:
            maps.append(cnn(x))
            
        maps = torch.stack(maps)
        maps = self.relu(maps)
        
        return maps