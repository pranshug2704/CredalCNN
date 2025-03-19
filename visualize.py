import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import argparse
from CredalCNN import CredalCNN

# Check for output directory
output_dir = 'visualizations/original_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CredalCNN().to(device)

# Check if the model file exists
model_path = 'credal_cnn_model.pth'
improved_model_path = None

# Look for improved models
if os.path.exists('models'):
    improved_models = [f for f in os.listdir('models') if f.endswith('.pth')]
    if improved_models:
        # Use the most recent model based on timestamp
        improved_models.sort(reverse=True)
        improved_model_path = os.path.join('models', improved_models[0])

if improved_model_path and os.path.exists(improved_model_path):
    # Load an improved model (has a different format)
    checkpoint = torch.load(improved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint.get('accuracy', 'unknown')
    print(f"Improved model loaded from {improved_model_path}")
    print(f"Model accuracy: {accuracy}%")
    # Update output directory to match the model
    output_dir = f'visualizations/improved_model_{os.path.basename(improved_model_path).split(".")[0]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
elif os.path.exists(model_path):
    # Load the original model
    model.load_state_dict(torch.load(model_path))
    print(f"Original model loaded from {model_path}")
else:
    print(f"No model files found. Please run CredalCNN.py or ImprovedCredalCNN.py first.")
    exit(1)

model.eval()

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Get class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to calculate uncertainty as difference between upper and lower bounds
def get_uncertainty(lower_bound, upper_bound):
    return torch.mean(upper_bound - lower_bound, dim=1)

# Collect predictions, ground truth, and uncertainty
all_uncertainties = []
all_predicted = []
all_labels = []
all_correct = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        lower_bound, upper_bound = model(images)
        
        # Apply softmax to normalize bounds
        lower_bound = torch.nn.functional.softmax(lower_bound, dim=1)
        upper_bound = torch.nn.functional.softmax(upper_bound, dim=1)
        
        # Calculate uncertainty (difference between upper and lower bounds)
        uncertainty = get_uncertainty(lower_bound, upper_bound)
        
        # Use average of bounds for prediction
        predictions = (lower_bound + upper_bound) / 2
        _, predicted = torch.max(predictions, 1)
        
        correct = (predicted == labels)
        
        all_uncertainties.extend(uncertainty.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_correct.extend(correct.cpu().numpy())

# Convert to numpy arrays
all_uncertainties = np.array(all_uncertainties)
all_predicted = np.array(all_predicted)
all_labels = np.array(all_labels)
all_correct = np.array(all_correct)

# 1. Plot uncertainty distribution for correct vs incorrect predictions
plt.figure(figsize=(10, 6))
plt.hist(all_uncertainties[all_correct], bins=50, alpha=0.5, label='Correct predictions')
plt.hist(all_uncertainties[~all_correct], bins=50, alpha=0.5, label='Incorrect predictions')
plt.xlabel('Uncertainty (Upper - Lower Bound Difference)')
plt.ylabel('Count')
plt.title('Uncertainty Distribution: Correct vs Incorrect Predictions')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/uncertainty_distribution.png')
plt.close()

# 2. Class-wise accuracy
class_correct = np.zeros(10)
class_total = np.zeros(10)

for i in range(len(all_labels)):
    label = all_labels[i]
    class_correct[label] += all_correct[i]
    class_total[label] += 1

class_accuracy = class_correct / class_total

plt.figure(figsize=(12, 6))
plt.bar(classes, class_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/class_accuracy.png')
plt.close()

# 3. Class-wise uncertainty
class_uncertainty = np.zeros(10)
for i in range(len(all_labels)):
    label = all_labels[i]
    class_uncertainty[label] += all_uncertainties[i]

class_avg_uncertainty = class_uncertainty / class_total

plt.figure(figsize=(12, 6))
plt.bar(classes, class_avg_uncertainty)
plt.xlabel('Class')
plt.ylabel('Average Uncertainty')
plt.title('Class-wise Average Uncertainty')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/class_uncertainty.png')
plt.close()

# 4. Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_predicted)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png')
plt.close()

# 5. Uncertainty vs. Accuracy
# Divide uncertainty into bins and calculate accuracy for each bin
uncertainty_bins = np.linspace(min(all_uncertainties), max(all_uncertainties), 10)
bin_indices = np.digitize(all_uncertainties, uncertainty_bins)
bin_accuracy = np.zeros(len(uncertainty_bins)+1)
bin_counts = np.zeros(len(uncertainty_bins)+1)

for i in range(len(all_correct)):
    bin_idx = bin_indices[i]
    bin_accuracy[bin_idx] += all_correct[i]
    bin_counts[bin_idx] += 1

# Avoid division by zero
bin_counts[bin_counts == 0] = 1
bin_accuracy = bin_accuracy / bin_counts

plt.figure(figsize=(10, 6))
plt.plot(uncertainty_bins, bin_accuracy[1:], marker='o')
plt.xlabel('Uncertainty Level')
plt.ylabel('Accuracy')
plt.title('Relationship Between Uncertainty and Accuracy')
plt.grid(True)
plt.savefig(f'{output_dir}/uncertainty_vs_accuracy.png')
plt.close()

print(f"Visualizations have been created and saved in {output_dir}")