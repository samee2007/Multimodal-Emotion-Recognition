import torch
from meld_dataset import MultimodalEmotionDataset
from text_model import TextModel
from audio_model import AudioModel
from visual_model import VisualModel
from fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel

def test_models():
    print("Testing Model Architectures with Dummy Data...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load a single batch of dummy data
    dataset = MultimodalEmotionDataset(num_samples=2)
    batch = dataset[0]
    
    # Add batch dimension
    input_ids = batch["text"]["input_ids"].unsqueeze(0).to(device)
    attention_mask = batch["text"]["attention_mask"].unsqueeze(0).to(device)
    audio = batch["audio"].unsqueeze(0).to(device)
    visual = batch["visual"].unsqueeze(0).to(device)
    
    print(f"Input Shapes:")
    print(f"  Text (input_ids): {input_ids.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Visual: {visual.shape}\n")
    
    # 2. Test Single Modality Models
    print("--- Single Modality Baselines ---")
    
    text_model = TextModel(fine_tune=False).to(device)
    out = text_model(input_ids, attention_mask)
    print(f"TextModel Output Shape: {out.shape}")
    
    audio_model = AudioModel(use_pretrained=False).to(device)
    out = audio_model(audio)
    print(f"AudioModel Output Shape: {out.shape}")
    
    visual_model = VisualModel(use_pretrained=False).to(device)
    out = visual_model(visual)
    print(f"VisualModel Output Shape: {out.shape}\n")
    
    # 3. Test Fusion Models
    print("--- Multimodal Fusion Models ---")
    
    early_fusion = EarlyFusionModel().to(device)
    # Using small initialized weights to avoid slow test
    out = early_fusion(input_ids, attention_mask, audio, visual)
    print(f"EarlyFusionModel Output Shape: {out.shape}")
    
    late_fusion = LateFusionModel().to(device)
    out = late_fusion(input_ids, attention_mask, audio, visual)
    print(f"LateFusionModel Output Shape: {out.shape}")
    
    attn_fusion = AttentionFusionModel().to(device)
    out = attn_fusion(input_ids, attention_mask, audio, visual)
    print(f"AttentionFusionModel Output Shape: {out.shape}")
    
    print("\nAll models passed the forward pass test successfully!")

if __name__ == "__main__":
    test_models()
