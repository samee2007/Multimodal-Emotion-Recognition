import torch
import torch.nn as nn
import torchvision.models as models

class AudioModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=7, use_pretrained=True):
        super(AudioModel, self).__init__()
        # Using ResNet-18 as the backbone for Mel-spectrograms
        # Spectrograms usually have 1 channel, so we modify the first conv layer
        # Handle the weights parameter correctly for modern torchvision
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Modify the first conv layer to accept 1 channel (grayscale) instead of 3 (RGB)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Initialize the new conv layer weights by averaging the original weights across RGB channels
        if use_pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))

        # Remove the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.fc1 = nn.Linear(num_ftrs, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x):
        # x shape: (batch_size, 1, height (e.g., n_mels), width (e.g., time_frames))
        features = self.resnet(x)
        features = self.dropout(self.relu(self.fc1(features)))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
