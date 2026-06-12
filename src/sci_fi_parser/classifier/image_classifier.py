from sci_fi_parser.classifier.cnn import CNNClassifier
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch

class ImageClassifier:
    def __init__(self, model: CNNClassifier = None, image_transform = None):
        if model is None:
            self.model = ImageClassifier.create_dummy_model()
        else:
            self.model = model
        if image_transform is None:
            self.transform = ImageClassifier.create_dummy_tranform()
        else:
            self.transform = image_transform

    def classify_image(self, image):
        with torch.no_grad(): #with a trained network, gradient computation is not needed
            with Image.open(image) as im:
                as_tensor = torch.unsqueeze(self.transform(im), 0) #A dummy batch dimension is added to the tensor
                model_output = self.model.forward(as_tensor)
                return torch.argmax(model_output), model_output
            
    @classmethod
    def create_dummy_model(cls):
        input_size = 128
        return CNNClassifier(num_classes=6, channels_in=3, channels_out=[32, 64, 128], 
                          conv_kernel_size=3, pool_size=2, linear_layer_neurons=256, input_image_size=input_size)

    @classmethod
    def create_dummy_tranform(cls):
        input_size = 128
        return transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
        
if __name__=="__main__":
    im_classifier = ImageClassifier()
    my_image = Path("cnn_data_split/val/graphs_val/0a14bb795a27.jpg")
    classification, full_output = im_classifier.classify_image(my_image)
    print(classification, full_output)
