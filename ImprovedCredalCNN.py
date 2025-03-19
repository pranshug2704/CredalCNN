import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import datetime

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Define the Improved Credal Set-based CNN Model
class CredalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CredalCNN, self).__init__()
        # Enhanced architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes * 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        lower_bound, upper_bound = torch.chunk(x, 2, dim=1)
        return lower_bound, upper_bound

# Define the Credal Loss Function (Worst-Case Probability Decision)
def credal_loss(lower_bound, upper_bound, target):
    batch_size = lower_bound.size(0)
    target_one_hot = torch.zeros_like(lower_bound)
    target_one_hot[range(batch_size), target] = 1.0
    
    # Apply softmax to ensure bounds are properly normalized
    lower_bound_softmax = torch.nn.functional.softmax(lower_bound, dim=1)
    upper_bound_softmax = torch.nn.functional.softmax(upper_bound, dim=1)
    
    # Ensure numerical stability with small epsilon
    epsilon = 1e-6
    lower_bound_softmax = torch.clamp(lower_bound_softmax, min=epsilon)
    upper_bound_softmax = torch.clamp(upper_bound_softmax, min=epsilon)
    
    # Use the lower bound for true class, upper bound for other classes
    worst_case_prob = lower_bound_softmax * target_one_hot + upper_bound_softmax * (1 - target_one_hot)
    
    # Calculate cross-entropy loss with the worst-case probabilities
    loss = -torch.mean(torch.sum(target_one_hot * torch.log(worst_case_prob), dim=1))
    return loss

def train_and_evaluate():
    # Training parameters
    batch_size = 128  # Increased batch size
    learning_rate = 0.001
    epochs = 30  # Increased epochs
    
    # Enhanced data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Standard transform for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 Dataset with augmentation
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize Model, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CredalCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Training Loop
    best_accuracy = 0.0
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            lower_bound, upper_bound = model(images)
            loss = credal_loss(lower_bound, upper_bound, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        
        # Evaluate after each epoch
        model.eval()
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
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Generate timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/improved_credal_cnn_{timestamp}_acc_{accuracy:.2f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': epoch_loss,
            }, model_path)
            print(f"New best model saved to {model_path}")
    
    # Save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f'models/improved_credal_cnn_final_{timestamp}_acc_{accuracy:.2f}.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': epoch_loss,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    
    return model, best_accuracy

if __name__ == "__main__":
    model, accuracy = train_and_evaluate()
    print("Training and evaluation complete!")