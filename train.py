import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from meld_dataset import MultimodalEmotionDataset
from text_model import TextModel
from audio_model import AudioModel
from visual_model import VisualModel
from fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel
from torch.utils.data import random_split
                    
def train_epoch(model, dataloader, criterion, optimizer, device, modality="fusion"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move inputs to device
        input_ids = batch["text"]["input_ids"].to(device)
        attention_mask = batch["text"]["attention_mask"].to(device)
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass based on modality type
        if modality == "text":
            outputs = model(input_ids, attention_mask)
        elif modality == "audio":
            outputs = model(audio)
        elif modality == "visual":
            outputs = model(visual)
        else: # fusion
            outputs = model(input_ids, attention_mask, audio, visual)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, modality="fusion"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["text"]["input_ids"].to(device)
            attention_mask = batch["text"]["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)
            
            if modality == "text":
                outputs = model(input_ids, attention_mask)
            elif modality == "audio":
                outputs = model(audio)
            elif modality == "visual":
                outputs = model(visual)
            else:
                outputs = model(input_ids, attention_mask, audio, visual)
                
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument("--modality", type=str, default="attention_fusion", 
                        choices=["text", "audio", "visual", "early_fusion", "late_fusion", "attention_fusion"],
                        help="Which modality/fusion model to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare data
    full_dataset = MultimodalEmotionDataset() # Real data from disk
    
    # Split into Train (70%), Val (15%), Test (15%)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Select model
    print(f"Initializing {args.modality} Model...")
    if args.modality == "text":
        model = TextModel(num_classes=7).to(device)
        train_modality = "text"
    elif args.modality == "audio":
        model = AudioModel(num_classes=7).to(device)
        train_modality = "audio"
    elif args.modality == "visual":
        model = VisualModel(num_classes=7).to(device)
        train_modality = "visual"
    elif args.modality == "early_fusion":
        model = EarlyFusionModel(num_classes=7).to(device)
        train_modality = "fusion"
    elif args.modality == "late_fusion":
        model = LateFusionModel(num_classes=7).to(device)
        train_modality = "fusion"
    elif args.modality == "attention_fusion":
        model = AttentionFusionModel(num_classes=7).to(device)
        train_modality = "fusion"
    
    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 4. Training loop
    num_epochs = args.epochs
    best_loss = float('inf')
    model_save_path = f"best_model_{args.modality}.pth"
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, modality=train_modality)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, modality=train_modality)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model checkpoint to '{model_save_path}'")
