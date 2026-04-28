import os
import json
import torch
import cv2
import librosa
import numpy as np
from transformers import AutoTokenizer

def extract_video_features(video_path, seq_length=15, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return torch.zeros((seq_length, 3, target_size[0], target_size[1]))

    indices = np.linspace(0, total_frames - 1, seq_length, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] and transpose to (C, H, W)
            frame = frame.transpose((2, 0, 1)) / 255.0
            frames.append(frame)
        else:
            # Fallback if frame read fails
            frames.append(np.zeros((3, target_size[0], target_size[1])))
            
    cap.release()
    
    # Convert to tensor: (seq_len, C, H, W)
    frames_tensor = torch.tensor(np.stack(frames), dtype=torch.float32)
    return frames_tensor

def extract_audio_features(audio_path, target_shape=(128, 128)):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0], hop_length=len(y)//target_shape[1] + 1)
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize/Pad to ensure exact shape (128, 128)
    if log_mel_spec.shape[1] > target_shape[1]:
        log_mel_spec = log_mel_spec[:, :target_shape[1]]
    elif log_mel_spec.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
    # Add channel dimension: (1, H, W)
    log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
    
    return torch.tensor(log_mel_spec, dtype=torch.float32)

def main():
    import csv
    
    csv_path = "data/raw/train_sent_emo.csv"
    video_dir = "data/real_meld_subset"
    output_path = "data/processed_real/dataset.pt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Parse annotations
    annotations = {}
    metadata_path = os.path.join(video_dir, "metadata.json")
    
    if os.path.exists(metadata_path):
        print("Using generated metadata.json annotations...")
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            for item in meta:
                annotations[item["id"] + ".mp4"] = {
                    "text": item["text"],
                    "emotion": item["emotion"].lower()
                }
    elif os.path.exists(csv_path):
        print("Using CSV annotations...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
                annotations[filename] = {
                    "text": row['Utterance'],
                    "emotion": row['Emotion'].lower()
                }
    else:
        print(f"Error: Neither metadata.json nor {csv_path} found.")
        return
        
    emotions_map = {'anger': 0, 'disgust': 1, 'sadness': 2, 'joy': 3, 'neutral': 4, 'surprise': 5, 'fear': 6}
    
    print("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    processed_data = []
    
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Processing {len(mp4_files)} real MELD samples...")
    
    for i, filename in enumerate(mp4_files):
        video_path = os.path.join(video_dir, filename)
        audio_path = video_path.replace(".mp4", ".wav")
        
        if filename not in annotations:
            print(f"Warning: Annotation missing for {filename}. Skipping.")
            continue
            
        text = annotations[filename]["text"]
        emotion_str = annotations[filename]["emotion"]
        if emotion_str not in emotions_map:
            print(f"Warning: Unknown emotion '{emotion_str}' for {filename}. Skipping.")
            continue
            
        label = emotions_map[emotion_str]
        
        # Extract audio track on the fly if it doesn't exist
        try:
            if not os.path.exists(audio_path):
                import subprocess
                import imageio_ffmpeg
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"Audio extraction failed for {filename}: {e}")
            continue

        try:
            # Video features
            v_features = extract_video_features(video_path)
            
            # Audio features
            a_features = extract_audio_features(audio_path)
            
            # Text features
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            
            processed_data.append({
                "text": {
                    "input_ids": encoded['input_ids'].squeeze(0),
                    "attention_mask": encoded['attention_mask'].squeeze(0)
                },
                "audio": a_features,
                "visual": v_features,
                "label": label
            })
        except Exception as e:
            print(f"Feature extraction failed for {filename}: {e}")
            continue
            
        if (i+1) % 5 == 0:
            print(f"Processed {i+1}/{len(mp4_files)}...")
            
    if len(processed_data) > 0:
        print(f"Saving {len(processed_data)} processed samples to {output_path}...")
        torch.save(processed_data, output_path)
        print("Authentic Data Preprocessing complete!")
    else:
        print("Error: No data was processed.")

if __name__ == "__main__":
    main()
