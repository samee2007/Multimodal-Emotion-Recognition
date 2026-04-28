import torch
import torch.nn as nn
from transformers import RobertaModel

class TextModel(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=256, num_classes=7, fine_tune=True):
        super(TextModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Freeze or unfreeze RoBERTa layers
        for param in self.roberta.parameters():
            param.requires_grad = fine_tune
            
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def extract_features(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # We use the pooled output for classification tasks
        pooled_output = outputs.pooler_output
        features = self.dropout(self.relu(self.fc1(pooled_output)))
        return features

    def forward(self, input_ids, attention_mask):
        features = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(features)
        return logits
