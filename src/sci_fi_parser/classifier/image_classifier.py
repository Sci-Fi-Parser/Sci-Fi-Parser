from sci_fi_parser.classifier.cnn import CNNClassifier
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch
from transformers import EfficientNetForImageClassification

class ImageClassifier:
    def __init__(self, image_labels: list = None, model: CNNClassifier = None, image_transform = None):
        if image_labels is None:
            self.image_labels = ImageClassifier.create_dummy_labels()
        else:
            self.image_labels = image_labels
        if model is None:
            self.model = ImageClassifier.create_dummy_model()
        else:
            self.model = model
        if image_transform is None:
            self.transform = ImageClassifier.create_dummy_tranform()
        else:
            self.transform = image_transform

    def classify_image(self, image_path):
        with torch.no_grad(): #with a trained network, gradient computation is not needed
            with Image.open(image_path) as im:
                as_tensor = torch.unsqueeze(self.transform(im), 0) #A dummy batch dimension is added to the tensor
                model_output = self.model.forward(as_tensor)
                label_index = torch.argmax(model_output).item()
                scores = {label: val.item() for label, val in zip(self.image_labels, model_output[0])}
                label = self.image_labels[label_index]
                return label, scores[label]
            
    @classmethod
    def create_dummy_model(cls):
        input_size = 128
        return CNNClassifier(num_classes=6, channels_in=3, channels_out=[32, 64, 128], 
                          conv_kernel_size=3, pool_size=2, linear_layer_neurons=256, input_image_size=input_size)

    @classmethod
    def create_dummy_tranform(cls):
        input_size = 128
        return transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
        
    @classmethod
    def create_dummy_labels(cls):
        return ["graphs_d", "graphs_h", "graphs_l", "graphs_s", "graphs_v", "graphs_val"]
    
class DoclingClassifier(ImageClassifier):
    def __init__(self):
        model_id = "docling-project/DocumentFigureClassifier-v2.5"
        self.model = EfficientNetForImageClassification.from_pretrained(model_id) 
        self.model.eval()

        self.image_labels = self.model.config.id2label
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163],
            ),
        ])
    

    def classify_image(self, image_path):
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            as_tensor = torch.unsqueeze(self.transform(im), 0)
            with torch.no_grad():
                logits = self.model(as_tensor).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        score = probs[0, pred_id].item()
        label = self.image_labels[pred_id]
        return label, score

if __name__=="__main__":
    classifier = DoclingClassifier()
    im_path = "cnn_data_split/val/graphs_val/0a14bb795a27.jpg"
    print(classifier.classify_image(im_path))