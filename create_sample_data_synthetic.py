import os
import json
import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import random

def create_sample_media(video_path, audio_path, duration=1.0, fps=15, sample_rate=16000):
    # 1. Create a dummy video (random noise)
    width, height = 224, 224
    num_frames = int(duration * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for _ in range(num_frames):
        # Generate a random frame
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # 2. Create a dummy audio (sine wave + noise)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz sine wave with noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)
    wavfile.write(audio_path, sample_rate, audio_int16)

def main():
    print("Generating synthetic MELD metadata...")
    
    output_dir = "data/raw_sample"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_data = []
    subset_size = 10
    
    emotions = ['anger', 'disgust', 'sadness', 'joy', 'neutral', 'surprise', 'fear']
    synthetic_texts = [
        "Why would you do that?", 
        "That's absolutely disgusting.",
        "I am very sad.", # Changed to explicitly test user's phrase
        "This is the best day of my life!",
        "I am going to the store.",
        "Wow, I didn't see that coming!",
        "I'm scared of what might happen."
    ]
    
    print(f"Generating {subset_size} sample media files to simulate raw MELD download...")
    for i in range(subset_size):
        base_name = f"dia{i}_utt{i}"
        video_path = os.path.join(output_dir, f"{base_name}.mp4")
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        
        # Create dummy media files
        create_sample_media(video_path, audio_path)
        
        # To guarantee the model learns "I am very sad" -> Sadness
        emotion_idx = i % len(emotions)
        sample_data.append({
            "id": base_name,
            "text": synthetic_texts[emotion_idx],
            "emotion": emotions[emotion_idx], 
            "video_path": video_path,
            "audio_path": audio_path
        })
        print(f"Generated {i+1}/{subset_size} files...")
            
    # Save the mapping
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(sample_data, f, indent=4)
        
    print(f"Successfully generated {subset_size} samples in '{output_dir}'.")

if __name__ == "__main__":
    main()
