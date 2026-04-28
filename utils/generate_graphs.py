import matplotlib.pyplot as plt
import numpy as np

# 4.1 Training vs Validation Loss Graph
epochs = [1, 2, 3, 4, 5]
train_loss = [1.84, 1.51, 1.22, 1.01, 0.92]
val_loss = [1.46, 1.45, 1.38, 1.33, 1.26]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss', color='blue')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss', color='orange')
plt.title('Training vs Validation Loss (Attention Fusion)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('loss_graph.png', bbox_inches='tight', dpi=300)
plt.close()

# 4.2 Accuracy Comparison Graph
models = ['Text-Only', 'Audio-Only', 'Video-Only', 'Early Fusion', 'Attention Fusion']
accuracies = [68.5, 62.1, 58.4, 70.2, 74.3] # Sample realistic data

plt.figure(figsize=(9, 5))
bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Accuracy Comparison Across Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold')

plt.savefig('accuracy_graph.png', bbox_inches='tight', dpi=300)
plt.close()

# 4.4 F1-Score Performance Comparison
f1_scores = [0.65, 0.59, 0.55, 0.68, 0.72] # Sample realistic data

plt.figure(figsize=(9, 5))
bars = plt.bar(models, f1_scores, color=['#17becf', '#bcbd22', '#7f7f7f', '#e377c2', '#8c564b'])
plt.title('F1-Score Comparison Across Models')
plt.ylabel('F1-Score')
plt.ylim(0, 1.0)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval}", ha='center', va='bottom', fontweight='bold')

plt.savefig('f1_score_graph.png', bbox_inches='tight', dpi=300)
plt.close()

print("Successfully generated loss_graph.png, accuracy_graph.png, and f1_score_graph.png!")
