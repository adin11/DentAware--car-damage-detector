from torch import nn
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F

class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
num_classes = len(class_names)
trained_model = None

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet(num_classes)
        trained_model.load_state_dict(torch.load("model/saved_model.pth", map_location=torch.device("cpu")))

        trained_model.eval()

    with torch.no_grad():
        outputs = trained_model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

        predicted_class = class_names[top_idx.item()]
        confidence = round(top_prob.item() * 100, 2)

        return {"class": predicted_class, "confidence": confidence}
