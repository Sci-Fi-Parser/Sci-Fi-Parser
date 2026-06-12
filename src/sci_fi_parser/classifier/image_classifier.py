from sci_fi_parser.classifier.cnn import CNNClassifier
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch

class ImageClassifier:
    def __init__(self, model: CNNClassifier, image_transform = None):
        if image_transform is not None:
            self.add_transform(image_transform)
        self.model = model

    def add_transform(self, image_transform):
        self.transform = image_transform

    def classify_image(self, image):
        with Image.open(image) as im:
            as_tensor = torch.unsqueeze(self.transform(im), 0) #A dummy batch dimension is added to the tensor
            return self.model.forward(as_tensor)

