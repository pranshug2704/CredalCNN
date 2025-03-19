import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the Credal Set-based CNN Model
class CredalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CredalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes * 2)  # Output lower and upper bounds
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        lower_bound, upper_bound = torch.chunk(x, 2, dim=1)  # Split into lower and upper bounds
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

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize Model, Optimizer, and Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CredalCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
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
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

print("Training complete.")

# Evaluate the model
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

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), 'credal_cnn_model.pth')
print("Model saved to credal_cnn_model.pth")
