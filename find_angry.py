import os, csv
subset = [f.replace('.mp4','') for f in os.listdir('data/real_meld_subset') if f.endswith('.mp4')]
reader = csv.DictReader(open('data/raw/train_sent_emo.csv', 'r', encoding='utf-8'))
angry_files = [(row['Dialogue_ID'], row['Utterance_ID'], row['Utterance']) for row in reader if row['Emotion'].lower()=='anger' and f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}" in subset]
print("ANGRY FILES:", angry_files)
