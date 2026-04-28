import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from meld_dataset import MultimodalEmotionDataset
from text_model import TextModel
from audio_model import AudioModel
from visual_model import VisualModel
from fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel

def evaluate_model(model, dataloader, device, modality="fusion"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["text"]["input_ids"].to(device)
            attention_mask = batch["text"]["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].numpy()
            
            # Forward pass
            if modality == "text":
                outputs = model(input_ids, attention_mask)
            elif modality == "audio":
                outputs = model(audio)
            elif modality == "visual":
                outputs = model(visual)
            else: # fusion
                outputs = model(input_ids, attention_mask, audio, visual)
                
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, precision, recall, f1, cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Model")
    parser.add_argument("--modality", type=str, default="attention_fusion", 
                        choices=["text", "audio", "visual", "early_fusion", "late_fusion", "attention_fusion"],
                        help="Which modality/fusion model to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing {args.modality} model for evaluation...")
    if args.modality == "text":
        model = TextModel(num_classes=7)
        eval_modality = "text"
    elif args.modality == "audio":
        model = AudioModel(num_classes=7)
        eval_modality = "audio"
    elif args.modality == "visual":
        model = VisualModel(num_classes=7)
        eval_modality = "visual"
    elif args.modality == "early_fusion":
        model = EarlyFusionModel(num_classes=7)
        eval_modality = "fusion"
    elif args.modality == "late_fusion":
        model = LateFusionModel(num_classes=7)
        eval_modality = "fusion"
    elif args.modality == "attention_fusion":
        model = AttentionFusionModel(num_classes=7)
        eval_modality = "fusion"
    
    model_path = f"best_model_{args.modality}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Could not load {model_path}. Make sure you run train.py --modality {args.modality} first! Error: {e}")
        import sys; sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # 2. Load authentic Dataset and apply exact same split as train.py
    print("Loading dataset for evaluation...")
    from torch.utils.data import random_split
    try:
        full_dataset = MultimodalEmotionDataset(data_path="data/processed_real/dataset.pt")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import sys; sys.exit(1)
        
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Critical: Use same manual_seed to ensure same split as train.py
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Evaluate
    acc, precision, recall, f1, cm = evaluate_model(model, test_loader, device, modality=eval_modality)
    
    print(f"\n--- Evaluation Results ({args.modality}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Emotion classes (e.g., from MELD)
    emotion_classes = ['Anger', 'Disgust', 'Sadness', 'Joy', 'Neutral', 'Surprise', 'Fear']
    
    # Save confusion matrix with modality name
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {args.modality}')
    plt.savefig(f'confusion_matrix_{args.modality}.png')
    print(f"Saved confusion matrix to 'confusion_matrix_{args.modality}.png'")
