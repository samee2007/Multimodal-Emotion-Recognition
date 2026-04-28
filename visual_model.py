import torch
import torch.nn as nn
import torchvision.models as models

class VisualModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=7, use_pretrained=True):
        super(VisualModel, self).__init__()
        
        # ResNet-50 feature extractor for individual frames
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the classification layer
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.cnn = resnet
        
        # LSTM to process the sequence of frame features
        self.lstm = nn.LSTM(input_size=self.feature_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
                            
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process all frames through the CNN
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)
        
        # Reshape back to sequence
        cnn_features = cnn_features.view(batch_size, seq_len, self.feature_dim)
        
        # Process through LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        features = self.dropout(self.relu(self.fc1(last_out)))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
