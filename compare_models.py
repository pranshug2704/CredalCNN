import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import torch
from CredalCNN import CredalCNN as OriginalCredalCNN
from ImprovedCredalCNN import CredalCNN as ImprovedCredalCNN
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_model_accuracy(model_path):
    # Extract accuracy from filename if available
    match = re.search(r'acc_([0-9.]+)', model_path)
    if match:
        return float(match.group(1))
    
    # Otherwise run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine which model class to use
    if 'improved' in model_path:
        model = ImprovedCredalCNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = OriginalCredalCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Load CIFAR-10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            lower_bound, upper_bound = model(images)
            
            # Apply softmax to normalize bounds
            lower_bound = torch.nn.functional.softmax(lower_bound, dim=1)
            upper_bound = torch.nn.functional.softmax(upper_bound, dim=1)
            
            # Use average of lower and upper bounds for prediction
            predictions = (lower_bound + upper_bound) / 2
            _, predicted = torch.max(predictions, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def compare_models():
    # Create output directory
    output_dir = 'comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all models
    original_model_path = 'credal_cnn_model.pth'
    improved_models = sorted(glob.glob('models/*.pth'))
    
    if not os.path.exists(original_model_path):
        print(f"Original model not found at {original_model_path}")
        return
    
    if not improved_models:
        print("No improved models found in 'models/' directory")
        return
    
    # Get accuracies
    print("Calculating model accuracies...")
    original_accuracy = get_model_accuracy(original_model_path)
    
    improved_accuracies = []
    model_names = []
    
    for model_path in improved_models:
        model_name = os.path.basename(model_path).split('.')[0]
        model_names.append(model_name)
        print(f"Evaluating {model_name}...")
        accuracy = get_model_accuracy(model_path)
        improved_accuracies.append(accuracy)
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Create accuracy comparison chart
    plt.figure(figsize=(12, 6))
    
    # Add original model
    plt.bar(0, original_accuracy, color='blue', label='Original Model')
    plt.text(0, original_accuracy + 1, f"{original_accuracy:.2f}%", ha='center')
    
    # Add improved models
    for i, (name, accuracy) in enumerate(zip(model_names, improved_accuracies)):
        plt.bar(i+1, accuracy, color='green')
        plt.text(i+1, accuracy + 1, f"{accuracy:.2f}%", ha='center')
    
    plt.axhline(y=original_accuracy, color='red', linestyle='--', label='Original Model Baseline')
    
    # Set chart labels
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(range(len(improved_models) + 1), ['Original'] + [f'Improved {i+1}' for i in range(len(improved_models))])
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(f'{output_dir}/accuracy_comparison.png')
    plt.close()
    
    print(f"Comparison chart saved to {output_dir}/accuracy_comparison.png")
    
    # Print results summary
    print("\n===== MODEL COMPARISON SUMMARY =====")
    print(f"Original Model: {original_accuracy:.2f}%")
    
    best_accuracy = max(improved_accuracies) if improved_accuracies else 0
    best_index = improved_accuracies.index(best_accuracy) if improved_accuracies else -1
    
    for i, (name, accuracy) in enumerate(zip(model_names, improved_accuracies)):
        improvement = accuracy - original_accuracy
        print(f"Improved Model {i+1}: {accuracy:.2f}% ({'+' if improvement >= 0 else ''}{improvement:.2f}%)")
    
    if best_index >= 0:
        print(f"\nBest Model: Improved Model {best_index+1} with {best_accuracy:.2f}% accuracy")
        print(f"Improvement over original: {best_accuracy - original_accuracy:.2f}%")

if __name__ == "__main__":
    compare_models()