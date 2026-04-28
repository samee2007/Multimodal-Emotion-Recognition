import torch
import torch.nn as nn
from text_model import TextModel
from audio_model import AudioModel
from visual_model import VisualModel

class EarlyFusionModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=7):
        super(EarlyFusionModel, self).__init__()
        self.text_model = TextModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.audio_model = AudioModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.visual_model = VisualModel(hidden_dim=hidden_dim, num_classes=num_classes)
        
        # In early fusion, we concatenate the extracted features before classification
        # Since each modality extracts a vector of size `hidden_dim`, the concatenated size is 3 * hidden_dim
        self.fusion_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, audio_x, visual_x):
        text_features = self.text_model.extract_features(input_ids, attention_mask)
        audio_features = self.audio_model.extract_features(audio_x)
        visual_features = self.visual_model.extract_features(visual_x)
        
        # Concatenate features: (batch_size, 3 * hidden_dim)
        fused_features = torch.cat((text_features, audio_features, visual_features), dim=1)
        
        # Process through fusion layer
        x = self.dropout(self.relu(self.fusion_fc(fused_features)))
        logits = self.classifier(x)
        
        return logits

class LateFusionModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=7):
        super(LateFusionModel, self).__init__()
        self.text_model = TextModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.audio_model = AudioModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.visual_model = VisualModel(hidden_dim=hidden_dim, num_classes=num_classes)
        
        # Learnable weights for each modality
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, input_ids, attention_mask, audio_x, visual_x):
        # Get logits from individual models
        text_logits = self.text_model(input_ids, attention_mask)
        audio_logits = self.audio_model(audio_x)
        visual_logits = self.visual_model(visual_x)
        
        # Normalize weights using softmax to ensure they sum to 1
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # Combine logits using learned weights
        fused_logits = (normalized_weights[0] * text_logits + 
                        normalized_weights[1] * audio_logits + 
                        normalized_weights[2] * visual_logits)
                        
        return fused_logits

class AttentionFusionModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=7, num_heads=4):
        super(AttentionFusionModel, self).__init__()
        self.text_model = TextModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.audio_model = AudioModel(hidden_dim=hidden_dim, num_classes=num_classes)
        self.visual_model = VisualModel(hidden_dim=hidden_dim, num_classes=num_classes)
        
        # Multihead Attention expects queries, keys, and values.
        # We will let text be the query, and audio/video be the key/values
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, audio_x, visual_x):
        text_features = self.text_model.extract_features(input_ids, attention_mask)
        audio_features = self.audio_model.extract_features(audio_x)
        visual_features = self.visual_model.extract_features(visual_x)
        
        # Reshape to (batch_size, seq_len, hidden_dim) where seq_len is 1 for embeddings
        text_features = text_features.unsqueeze(1)
        audio_features = audio_features.unsqueeze(1)
        visual_features = visual_features.unsqueeze(1)
        
        # Key/Values: concatenate audio and visual features
        # Shape: (batch_size, 2, hidden_dim)
        kv_features = torch.cat((audio_features, visual_features), dim=1)
        
        # Query: Text features. We see which parts of audio/visual are most relevant to the text
        attn_output, _ = self.cross_attention(query=text_features, key=kv_features, value=kv_features)
        
        # Remove sequence dimension: (batch_size, hidden_dim)
        attn_output = attn_output.squeeze(1)
        
        # Add residual connection
        fused_features = attn_output + text_features.squeeze(1)
        
        logits = self.fc(fused_features)
        return logits
