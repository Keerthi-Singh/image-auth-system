import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pretrained ResNet model
base_model = models.resnet18(pretrained=True)

# Remove classification layer → feature extractor
feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
feature_extractor.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_embedding(image_path):
    """
    Generate feature embedding (digital fingerprint)
    """
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = feature_extractor(img)

    return embedding.flatten().numpy()

def classify_image(image_path):
    """
    Basic classification using pretrained model
    """
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = base_model(img)

    _, predicted = torch.max(outputs, 1)
    return predicted.item()