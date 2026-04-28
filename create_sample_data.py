import os
import json
import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import requests
import csv

def create_sample_media(video_path, audio_path, duration=2.0, fps=15, sample_rate=16000):
    # 1. Create a dummy video (black frames with some random noise)
    width, height = 224, 224
    num_frames = int(duration * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # 2. Create a dummy audio (sine wave + noise)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    audio_int16 = np.int16(audio * 32767)
    wavfile.write(audio_path, sample_rate, audio_int16)

def main():
    output_dir = "data/real_meld_subset"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "train_sent_emo.csv")
    if not os.path.exists(csv_path):
        print("Downloading just the tiny CSV annotations...")
        url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
        response = requests.get(url)
        with open(csv_path, "wb") as f:
            f.write(response.content)
            
    print("Reading CSV to generate synthetic media...")
    sample_data = []
    subset_size = 50
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= subset_size:
                break
                
            base_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            video_path = os.path.join(output_dir, f"{base_name}.mp4")
            audio_path = os.path.join(output_dir, f"{base_name}.wav")
            
            # Create dummy media files
            create_sample_media(video_path, audio_path)
            
            sample_data.append({
                "id": base_name,
                "text": row['Utterance'],
                "emotion": row['Emotion'],
                "video_path": video_path,
                "audio_path": audio_path
            })
            if (i+1) % 10 == 0:
                print(f"Generated {i+1}/{subset_size} files...")
                
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(sample_data, f, indent=4)
        
    print(f"Successfully generated {subset_size} samples in '{output_dir}'.")

if __name__ == "__main__":
    main()
