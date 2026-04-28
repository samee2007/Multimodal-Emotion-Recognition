import os
import requests
from tqdm import tqdm
import tarfile

def download_file(url, dest_path):
    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    print(f"Downloading {url} to {dest_path}")
    print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
    
    with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    os.makedirs("data/raw", exist_ok=True)
    
    csv_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
    csv_path = "data/raw/train_sent_emo.csv"
    if not os.path.exists(csv_path):
        download_file(csv_url, csv_path)
    
    # Huggingface dataset mirror for MELD raw video
    tar_url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    tar_path = "data/MELD.Raw.tar.gz"
    
    if not os.path.exists(tar_path):
        download_file(tar_url, tar_path)
        
    print("Download complete. Extracting the archive...")
    
    # Extract
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path="data/raw/")
        
    print("Extraction complete! You now have the REAL MELD dataset.")
    
if __name__ == "__main__":
    main()
