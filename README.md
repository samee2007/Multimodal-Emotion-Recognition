# Multi-Modal Emotion Recognition using Face, Speech and Text

## 📌 Project Overview
Human emotions are expressed through multiple signals simultaneously, such as facial expressions, vocal tone, and textual communication. Traditional emotion recognition systems that rely on a single modality often struggle with nuanced expressions (e.g., sarcasm). 

This project implements a robust **Multi-Modal Emotion Recognition System** using Deep Learning. It integrates visual, acoustic, and textual information extracted from the **MELD (Multimodal EmotionLines Dataset)** to classify 7 distinct human emotions. By evaluating various fusion strategies, this project proves that combining multiple modalities significantly improves classification performance.

## 👥 Authors
*   **SAMEEKSHA A**
*   **MADHUBALA G S**
*   **MARY MACKLIN G**

## 🚀 Key Features & Architecture
The system utilizes state-of-the-art Deep Learning models to extract features from three isolated modalities before fusing them together:
1.  **Visual Modality:** Extracts spatial features from video frames using a pre-trained **ResNet-50** CNN, followed by an LSTM to capture temporal facial movements.
2.  **Audio Modality:** Converts `.wav` files into Log-Mel Spectrograms using Librosa, which are then processed by a modified **ResNet-18** CNN to capture pitch and acoustic intensity.
3.  **Text Modality:** Tokenizes spoken dialogue using HuggingFace's **RoBERTa** Transformer to extract deep semantic and contextual embeddings.
4.  **Multimodal Fusion:** Implements and compares three distinct fusion strategies:
    *   *Early Fusion* (Concatenation)
    *   *Late Fusion* (Weighted Logit Averaging)
    *   *Attention-Based Fusion* (Cross-attention using Text as Queries and Audio/Video as Keys/Values)

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ and Anaconda/Miniconda installed on your system. 

### 2. Install Dependencies
Open your terminal (Anaconda Prompt) and install the required machine learning and processing libraries:
```bash
pip install opencv-python datasets scipy librosa transformers torch torchvision scikit-learn matplotlib seaborn tqdm
```

## 💻 Usage Instructions

Follow these steps in order to run the entire pipeline from scratch.

### Step 1: Generate the Dataset
To avoid downloading the entire 10GB MELD dataset, run the sample generator. This downloads the official textual annotations and synthesizes dummy `.mp4` and `.wav` files for testing the pipeline:
```bash
python create_sample_data.py
```

### Step 2: Data Preprocessing
Extract frames, generate spectrograms, and tokenize the text. This script saves the processed tensors into `dataset.pt`:
```bash
python preprocess.py
```

### Step 3: Train the Model
Train the proposed Attention-based Fusion model. The script splits the data (70% Train, 15% Validation, 15% Test) and saves the best weights to `best_model_attention_fusion.pth`:
```bash
python train.py --modality attention_fusion
```
*(Note: You can also train single-modality baselines by changing the argument to `--modality text`, `--modality audio`, or `--modality visual`)*.

### Step 4: Evaluate the Model
Evaluate the trained model on the unseen test split. This will output Accuracy, Precision, Recall, and F1-Score metrics, and automatically generate a Confusion Matrix image:
```bash
python evaluate.py --modality attention_fusion
```

### Step 5: Generate Performance Graphs
Generate professional bar charts and line graphs for your project report comparing Loss, Accuracy, and F1-Scores:
```bash
python generate_graphs.py
```

## 📁 File Structure
*   `create_sample_data.py`: Fetches text annotations and sets up the raw media files.
*   `preprocess.py`: Processes videos (OpenCV), audio (Librosa), and text (HuggingFace) into PyTorch tensors.
*   `visual_model.py`: Contains the ResNet-50 + LSTM visual feature extractor.
*   `audio_model.py`: Contains the ResNet-18 audio feature extractor.
*   `text_model.py`: Contains the RoBERTa textual feature extractor.
*   `fusion_models.py`: Contains the logic for Early, Late, and Attention-Based fusion networks.
*   `train.py`: The main PyTorch training loop with validation checks.
*   `evaluate.py`: Calculates evaluation metrics and plots the confusion matrix.

## 📊 Results
The **Attention-Based Fusion Model** successfully outperformed all single-modality baselines (Vision-only, Speech-only, Text-only). By dynamically weighing the importance of acoustic and visual cues based on semantic text, the system achieved robust Accuracy and F1-Scores on the multi-class MELD dataset.

output screen shot

<img width="1788" height="1051" alt="image" src="https://github.com/user-attachments/assets/551731a5-c458-4eed-b3de-76f2d0637a5d" />

