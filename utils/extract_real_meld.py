import os
import tarfile

def extract_subset(tar_path, output_dir, max_files=50):
    os.makedirs(output_dir, exist_ok=True)
    extracted = 0
    print(f"Opening archive: {tar_path}")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            # Look for video files inside the tar
            if member.isfile() and member.name.endswith(".mp4"):
                # We want to extract it directly into the output_dir without nested folders
                member.name = os.path.basename(member.name)
                tar.extract(member, path=output_dir)
                extracted += 1
                print(f"Extracted [{extracted}/{max_files}]: {member.name}")
                
                if extracted >= max_files:
                    print(f"Reached subset limit of {max_files} files.")
                    break
                    
    print(f"Successfully extracted {extracted} videos to {output_dir}")

if __name__ == "__main__":
    train_tar = "data/MELD.Raw/MELD.Raw/train.tar.gz"
    output_dir = "data/real_meld_subset"
    
    if not os.path.exists(train_tar):
        print(f"Error: Could not find {train_tar}.")
        print("Please ensure you extracted MELD.Raw.tar.gz properly.")
    else:
        extract_subset(train_tar, output_dir, max_files=50)
