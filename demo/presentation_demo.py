import os
import torch
import warnings
import logging
from transformers import AutoTokenizer

# Suppress all annoying warnings and errors for a clean presentation!
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from preprocess import extract_audio_features, extract_video_features
from fusion_models import AttentionFusionModel
from meld_dataset import MultimodalEmotionDataset
from evaluate import evaluate_model
from torch.utils.data import DataLoader, random_split

def presentation_run():
    print("="*60)
    print("🚀 MULTI-MODAL EMOTION RECOGNITION SYSTEM")
    print("="*60 + "\n")

    # Load Model globally for eval and inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionFusionModel(num_classes=7)
    try:
        model.load_state_dict(torch.load("best_model_attention_fusion.pth", map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    # 1. Show Data Split
    print("[1] LOADING DATASET...")
    try:
        full_dataset = MultimodalEmotionDataset(data_path="data/processed_real/dataset.pt")
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        print(f"    Total dataset size: {len(full_dataset)} Multimodal Samples (Video + Audio + Text)")
        print(f"    -> Training split   (70%): {train_size} samples")
        print(f"    -> Validation split (15%): {val_size} samples")
        print(f"    -> Testing split    (15%): {test_size} samples")
        print("    Status: Loaded successfully.\n")

        # 2. Show Evaluation Metrics
        print("[2] EVALUATING MODEL ON TEST SET...")
        print("    Running Attention-Based Fusion Model...")
        
        generator = torch.Generator().manual_seed(42)
        _, _, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        acc, precision, recall, f1, cm = evaluate_model(model, test_loader, device, modality="fusion")
        
        print("    --- Evaluation Results ---")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-score:  {f1:.4f}")
        print("    Status: Evaluation complete. Confusion matrix calculated.\n")

    except Exception as e:
        print(f"    Error during dataset loading or evaluation: {e}\n")

    # 3. Live Prediction (Inference)
    print("[3] LIVE PREDICTION ON RANDOM SAMPLE...")
    import json, random
    try:
        with open("data/real_meld_subset/metadata.json", "r") as f:
            metadata = json.load(f)
        random_sample = random.choice(metadata)
        sample_text = random_sample["text"]
        sample_vid = random_sample["video_path"]
        sample_aud = random_sample["audio_path"]
        ground_truth = random_sample["emotion"]
    except Exception as e:
        print(f"    Error loading metadata: {e}")
        return

    print(f"    Input Text:   '{sample_text}'")
    print(f"    Input Video:  {sample_vid}")
    print(f"    Input Audio:  {sample_aud}")
    print(f"    Ground Truth: ** {ground_truth.upper()} **")
    print("    Extracting features (ResNet & RoBERTa)....")

    try:
        # Extract
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        encoded = tokenizer(sample_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        audio_features = extract_audio_features(sample_aud).unsqueeze(0).to(device)
        video_features = extract_video_features(sample_vid).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(encoded['input_ids'].to(device), encoded['attention_mask'].to(device), audio_features, video_features)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        emotions = ['Anger', 'Disgust', 'Sadness', 'Joy', 'Neutral', 'Surprise', 'Fear']
        predicted_idx = torch.argmax(probabilities).item()
        predicted_emotion = emotions[predicted_idx]

        print("\n    >>> PREDICTION RESULT <<<")
        print(f"    Final Predicted Emotion: ** {predicted_emotion.upper()} **")
        print("    Confidence Scores:")
        for idx, (emo, prob) in enumerate(zip(emotions, probabilities)):
            marker = " <--" if idx == predicted_idx else ""
            print(f"      {emo:10s} : {prob.item():.4f}{marker}")

    except Exception as e:
        print(f"    Prediction Error: {e}")

    print("\n" + "="*60)
    print("✅ PRESENTATION RUN COMPLETE")
    print("="*60)

if __name__ == "__main__":
    presentation_run()
