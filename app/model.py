import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Common Layers
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layers
        self.linear1 = nn.Linear(64 * 16 * 32, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear4 = nn.Linear(1024, 512)

        # Output Layer
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.dropout(x)

        x = self.linear4(x)
        x = self.dropout(x)

        x = self.output(x)

        return x


def load_model(model_path, num_classes, device):
    """
    Load the trained model for inference.
    """

    model = Net(num_classes)

    checkpoint = torch.load(
    model_path,
    map_location=device
)


# If a full checkpoint was saved
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
    # If only model weights were saved
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model