import torch
import torch.nn as nn
import torch.nn.functional as F
from models import NoisyBase

class NoisyFeedForward(NoisyBase):
    def __init__(self, w_max=1, noise_spread=1, scale=1):
        super(NoisyFeedForward, self).__init__(w_max, noise_spread)
        self.scale = scale
        self.i1, self.n1, self.n2 = 28 * 28, 128 * self.scale, 64 * self.scale
        self.fc1 = nn.Linear(self.i1 * 2, self.n1,bias=False)
        self.bn1 = nn.BatchNorm1d(self.n1)
        self.fc2 = nn.Linear(self.n1 * 2, self.n2,bias=False)
        self.bn2 = nn.BatchNorm1d(self.n2)
        self.fc3 = nn.Linear(self.n2 * 2, 10,bias=False)
        self.ln = nn.LayerNorm(10)
        self.apply(self._init_weights)

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        x = torch.concat([x, -x], dim=-1)
        x = F.tanh(self.bn1(self.fc1(x)))
        x = torch.concat([x, -x], dim=-1)
        x = F.tanh(self.bn2(self.fc2(x)))
        x = torch.concat([x, -x], dim=-1)
        x = self.ln(self.fc3(x))
        return x


class NoisyResNet(NoisyBase):
    def __init__(self, num_classes=10, w_max=1, noise_spread=1, scale=1):
        super(NoisyResNet, self).__init__(w_max, noise_spread)

        self.scale = scale
        # Scale the number of channels
        base_channels = int(32 * scale)
        increased_channels = int(64 * scale)

        # First convolutional layer: convert input (after concatenation) to base_channels
        self.conv1 = nn.Conv2d(2, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual block: two layers that maintain base_channels
        self.conv2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)

        # Last convolutional block to increase the number of channels
        self.conv4 = nn.Conv2d(base_channels * 2, increased_channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(increased_channels)

        # Final fully connected layer: flatten and map to num_classes
        self.fc = nn.Linear(increased_channels * 7 * 7 * 2, num_classes,bias=False)
        self.ln = nn.LayerNorm(10)

        # Max pooling layer to reduce spatial dimensions (28x28 -> 14x14 and 14x14 -> 7x7)
        self.pool = nn.MaxPool2d(2, 2)

        # Initialize weights
        self.apply(self._init_weights)


    def forward(self, x):
        # First convolutional layer
        # Channels are doubled by concatenating x and -x to mimic negative weights using only positive ones
        x = torch.concat([x, -x], dim=1) * 1
        x = F.relu(self.bn1(self.conv1(x)))  # [batch, 32 * scale, 28, 28]

        # Residual block
        identity = x
        x = torch.concat([x, -x], dim=1)
        out = F.relu(self.bn2(self.conv2(x)))
        out = torch.concat([out, -out], dim=1)
        out = self.bn3(self.conv3(out))
        x = F.relu(out + identity)  # [batch, 32 * scale, 28, 28]

        # Downsampling with max pooling (28x28 -> 14x14)
        x = self.pool(x)  # [batch, 32 * scale, 14, 14]

        # Additional convolutional block
        x = torch.concat([x, -x], dim=1)
        x = F.relu(self.bn4(self.conv4(x)))  # [batch, 64 * scale, 14, 14]
        x = self.pool(x)  # [batch, 64 * scale, 7, 7]

        # Final fully connected layer
        x = x.view(x.size(0), -1)
        x = torch.concat([x, -x], dim=-1)
        x = self.ln(self.fc(x))

        return x

