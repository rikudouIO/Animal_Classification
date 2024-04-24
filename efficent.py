import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

import timm
import matplotlib.pyplot as plt


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AnimailDataSet(Dataset):
    def __init__(self, data, transform=None):
        self.data = ImageFolder(data, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
    

class AnimalClassifier(nn.Module):
    def __init__(self, classes=15):
        super(AnimalClassifier, self).__init__()

        self.model = timm.create_model('efficientnet_b0', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, classes)
        )

    def forward(self, x):
        features = self.model(x)
        outputs = self.classifier(features)
        return outputs
        
    
data_folder = 'animal_data'
batch_size = 32

model = AnimalClassifier(classes=15)
dataset = AnimailDataSet(data_folder, transform=data_transforms)

train_data = int(0.7 * len(dataset))
validation_data = int(0.15 * len(dataset))
test_data = len(dataset) - train_data - validation_data

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_data, validation_data, test_data]) # Devide images into 3 datasets

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)


print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(validation_dataset)} images")
print(f"Test set: {len(test_dataset)} images")

print("Classes:", dataset.classes)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
train_losses, validation_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

    validation_loss = running_loss / len(validation_loader.dataset)
    validation_losses.append(validation_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {validation_loss}")

    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {100 * accuracy:.2f}%')


plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
plt.plot(range(1, epoch+2), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()   


torch.save(model.state_dict(), 'model/animal_classifier_model.pth')