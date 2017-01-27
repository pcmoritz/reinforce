import torch.nn as nn

class VisionNet(nn.Module):
  def __init__(self, num_classes=10):
    super(VisionNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=8, stride=4),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 32, kernel_size=4, stride=2),
      nn.ReLU(inplace=True),
    )
    self.classifier = nn.Sequential(
      nn.Linear(32 * 8 * 8, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 32 * 8 * 8)
    x = self.classifier(x)
    return x
