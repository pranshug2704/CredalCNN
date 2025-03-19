import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ImprovedCredalCNN import CredalCNN
import argparse

# Function to calculate uncertainty as difference between upper and lower bounds
def get_uncertainty(lower_bound, upper_bound):
    return torch.mean(upper_bound - lower_bound, dim=1)

def evaluate_model(model_path, output_dir=None):
    # Create output directory for visualizations if it doesn't exist
    if output_dir is None:
        model_name = os.path.basename(model_path).split('.')[0]
        output_dir = f'visualizations/{model_name}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CredalCNN().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint.get('accuracy', 'unknown')
    model.eval()
    
    print(f"Evaluating model from {model_path}")
    print(f"Saved accuracy: {accuracy}%")
    
    # Load CIFAR-10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Collect predictions, ground truth, and uncertainty
    all_uncertainties = []
    all_predicted = []
    all_labels = []
    all_correct = []
    all_lower_bounds = []
    all_upper_bounds = []
    
    # Evaluate model
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            lower_bound, upper_bound = model(images)
            
            # Apply softmax to normalize bounds
            lower_bound_softmax = torch.nn.functional.softmax(lower_bound, dim=1)
            upper_bound_softmax = torch.nn.functional.softmax(upper_bound, dim=1)
            
            # Calculate uncertainty (difference between upper and lower bounds)
            uncertainty = get_uncertainty(lower_bound_softmax, upper_bound_softmax)
            
            # Use average of bounds for prediction
            predictions = (lower_bound_softmax + upper_bound_softmax) / 2
            _, predicted = torch.max(predictions, 1)
            
            correct_predictions = (predicted == labels)
            
            total += labels.size(0)
            correct += correct_predictions.sum().item()
            
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_correct.extend(correct_predictions.cpu().numpy())
            all_lower_bounds.append(lower_bound_softmax.cpu())
            all_upper_bounds.append(upper_bound_softmax.cpu())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Convert to numpy arrays
    all_uncertainties = np.array(all_uncertainties)
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    all_correct = np.array(all_correct)
    
    # Create visualizations
    
    # 1. Plot uncertainty distribution for correct vs incorrect predictions
    plt.figure(figsize=(10, 6))
    plt.hist(all_uncertainties[all_correct], bins=50, alpha=0.5, label='Correct predictions')
    plt.hist(all_uncertainties[~np.array(all_correct)], bins=50, alpha=0.5, label='Incorrect predictions')
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
    bars = plt.bar(range(10), class_accuracy, tick_label=classes)
    
    # Add accuracy values above bars
    for bar, accuracy in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.2f}', ha='center', va='bottom')
    
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
    bars = plt.bar(range(10), class_avg_uncertainty, tick_label=classes)
    
    # Add uncertainty values above bars
    for bar, uncertainty in zip(bars, class_avg_uncertainty):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                f'{uncertainty:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Average Uncertainty')
    plt.title('Class-wise Average Uncertainty')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_uncertainty.png')
    plt.close()
    
    # 4. Confusion Matrix
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
    valid_bins = bin_counts > 10  # Only use bins with significant data
    bin_accuracy = np.where(valid_bins, bin_accuracy / np.maximum(bin_counts, 1), 0)
    
    plt.figure(figsize=(10, 6))
    
    # Get center of bins for plotting
    bin_centers = uncertainty_bins
    
    # Only plot points with enough data
    valid_indices = valid_bins[1:]  # Shift to match bin_centers
    
    plt.plot(bin_centers[valid_indices], bin_accuracy[1:][valid_indices], marker='o', linestyle='-')
    plt.xlabel('Uncertainty Level')
    plt.ylabel('Accuracy')
    plt.title('Relationship Between Uncertainty and Accuracy')
    plt.grid(True)
    plt.savefig(f'{output_dir}/uncertainty_vs_accuracy.png')
    plt.close()
    
    # 6. Class Probability Distribution - Lower vs Upper bounds for each class
    all_lower_bounds = torch.cat(all_lower_bounds, dim=0).numpy()
    all_upper_bounds = torch.cat(all_upper_bounds, dim=0).numpy()
    
    # For each class, plot the average lower and upper bounds
    avg_lower_bounds = np.zeros(10)
    avg_upper_bounds = np.zeros(10)
    
    for c in range(10):
        indices = all_labels == c
        if np.sum(indices) > 0:  # If there are examples of this class
            class_lower_bounds = all_lower_bounds[indices]
            class_upper_bounds = all_upper_bounds[indices]
            
            # Average confidence for the true class
            avg_lower_bounds[c] = np.mean(class_lower_bounds[:, c])
            avg_upper_bounds[c] = np.mean(class_upper_bounds[:, c])
    
    # Plot bound differences
    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.35
    
    plt.bar(x - width/2, avg_lower_bounds, width, label='Lower Bound')
    plt.bar(x + width/2, avg_upper_bounds, width, label='Upper Bound')
    
    plt.xlabel('Class')
    plt.ylabel('Average Probability')
    plt.title('Average Probability Bounds for True Classes')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_probability_bounds.png')
    plt.close()
    
    # Return accuracy and visualizations path
    return accuracy, output_dir

def evaluate_all_models():
    # Find all model files
    model_files = glob.glob('models/*.pth')
    results = []
    
    for model_path in model_files:
        try:
            accuracy, vis_path = evaluate_model(model_path)
            results.append((model_path, accuracy, vis_path))
            print(f"Evaluated {model_path}, accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
    
    # Print summary
    print("\n===== MODEL EVALUATION SUMMARY =====")
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
    
    for i, (model_path, accuracy, vis_path) in enumerate(results):
        print(f"{i+1}. {os.path.basename(model_path)}: {accuracy:.2f}%")
    
    if results:
        best_model = results[0]
        print(f"\nBest model: {os.path.basename(best_model[0])} with accuracy {best_model[1]:.2f}%")
        print(f"Visualizations available at: {best_model[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CredalCNN models')
    parser.add_argument('--model', help='Path to specific model to evaluate (optional)')
    parser.add_argument('--all', action='store_true', help='Evaluate all models in the models directory')
    
    args = parser.parse_args()
    
    if args.model:
        evaluate_model(args.model)
    elif args.all:
        evaluate_all_models()
    else:
        # If no arguments, find the most recent model
        model_files = glob.glob('models/*.pth')
        if model_files:
            # Sort by modification time, newest first
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"Evaluating most recent model: {latest_model}")
            evaluate_model(latest_model)
        else:
            print("No models found. Please train a model first or specify a model path.")
            print("Usage: python evaluate_models.py --model <model_path>")
            print("       python evaluate_models.py --all")