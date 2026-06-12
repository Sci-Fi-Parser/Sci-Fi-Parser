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
        with torch.no_grad(): #with a trained network, gradient computation is not needed
            with Image.open(image) as im:
                as_tensor = torch.unsqueeze(self.transform(im), 0) #A dummy batch dimension is added to the tensor
                model_output = self.model.forward(as_tensor)
                return torch.argmax(model_output), model_output

        
if __name__=="__main__":
    input_size = 128
    model = CNNClassifier(num_classes=6, channels_in=3, channels_out=[32, 64, 128], 
                          conv_kernel_size=3, pool_size=2, linear_layer_neurons=256, input_image_size=input_size)
    im_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    im_classifier = ImageClassifier(model=model, image_transform=im_transform)
    my_image = Path("cnn_data_split/val/graphs_val/0a14bb795a27.jpg")
    classification, full_output = im_classifier.classify_image(my_image)
