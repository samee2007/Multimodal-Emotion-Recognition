import os
import csv
import json
import requests
import tarfile
from io import BytesIO

def download_meld_subset(target_count=50, output_dir="data/raw_meld"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download the CSV annotations from official MELD repo
    print("Downloading MELD annotations (train_sent_emo.csv)...")
    csv_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
    csv_path = os.path.join(output_dir, "train_sent_emo.csv")
    
    response = requests.get(csv_url)
    with open(csv_path, "wb") as f:
        f.write(response.content)
        
    # Read annotations into a dictionary for quick lookup
    # MELD CSV format: Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
    annotations = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The video file format is usually dia<Dialogue_ID>_utt<Utterance_ID>.mp4 or season/episode based
            # MELD Raw filenames are diaX_uttY.mp4
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            filename = f"dia{dia_id}_utt{utt_id}.mp4"
            annotations[filename] = {
                "text": row['Utterance'],
                "emotion": row['Emotion']
            }
            
    # 2. Stream the huge 10GB MELD.Raw.tar.gz and extract just the first `target_count` videos
    print(f"Streaming MELD.Raw.tar.gz from HuggingFace to extract {target_count} real videos...")
    print("This avoids downloading the full 10GB file to your hard drive!")
    
    tar_url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    
    metadata = []
    extracted_count = 0
    
    try:
        # Use a streaming request
        with requests.get(tar_url, stream=True) as r:
            r.raise_for_status()
            # Open tarfile from the raw stream
            # MELD.Raw.tar.gz -> MELD.Raw.tar -> .mp4 files
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".mp4"):
                        # Extract the filename from the path
                        filename = os.path.basename(member.name)
                        
                        # Only extract if we have annotations for it
                        if filename in annotations:
                            video_path = os.path.join(output_dir, filename)
                            audio_path = os.path.join(output_dir, filename.replace(".mp4", ".wav"))
                            
                            print(f"Extracting [{extracted_count+1}/{target_count}]: {filename}")
                            
                            # Extract the video file
                            f_out = open(video_path, "wb")
                            f_in = tar.extractfile(member)
                            if f_in:
                                f_out.write(f_in.read())
                            f_out.close()
                            
                            # Extract audio track using moviepy
                            try:
                                from moviepy.editor import VideoFileClip
                                clip = VideoFileClip(video_path)
                                clip.audio.write_audiofile(audio_path, logger=None)
                                clip.close()
                                
                                # Add to metadata
                                metadata.append({
                                    "id": filename.replace(".mp4", ""),
                                    "text": annotations[filename]["text"],
                                    "emotion": annotations[filename]["emotion"],
                                    "video_path": video_path,
                                    "audio_path": audio_path
                                })
                                extracted_count += 1
                                
                            except Exception as e:
                                print(f"Error processing {filename}: {e}")
                                
                            if extracted_count >= target_count:
                                print("Reached target count. Stopping stream.")
                                break
    except Exception as e:
        print(f"Streaming interrupted/finished: {e}")
        
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Successfully downloaded and processed {len(metadata)} REAL MELD samples!")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    download_meld_subset(target_count=10)
