import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import timm


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
    
    
model = AnimalClassifier()
model.load_state_dict(torch.load('model/animal_classifier_model.pth'))
model.eval()

image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_card_class(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class_index = torch.argmax(output, dim=1).item()
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

train_dataset = AnimailDataSet('animal_data', transform=image_transform)


image_path = 'animal_data/Panda/Panda_9_4.jpg' # image to predict
predicted_card_class = predict_card_class(image_path, model, image_transform, train_dataset.classes)

image = Image.open(image_path)
plt.imshow(image)
plt.title(f"Predicted class: {predicted_card_class}")
plt.axis('off')
plt.show()