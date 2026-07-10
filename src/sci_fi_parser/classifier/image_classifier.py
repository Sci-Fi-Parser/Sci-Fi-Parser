import torch
from PIL import Image, _typing
from torchvision import transforms
from transformers import EfficientNetForImageClassification

from typing import Optional, List
from pathlib import Path

from sci_fi_parser.classifier.cnn import CNNClassifier


class ImageClassifier:
    def __init__(self, image_labels: Optional[List] = None, model: Optional[CNNClassifier] = None, 
                 image_transform: Optional[transforms.transforms.Compose]=None):
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

    def classify_image(self, image_path: _typing.StrOrBytesPath) -> tuple[str|int, dict]:
        """
        Classifies a given image.
        inputs:
            image_path: path to the image
        outputs:
            a tuple containing the class assigned to the image and the confidence scores of each possible class

        """
        with torch.no_grad():  # with a trained network, gradient computation is not needed
            with Image.open(image_path) as im:
                as_tensor = torch.unsqueeze(
                    self.transform(im), 0
                )  # A dummy batch dimension is added to the tensor
                model_output = self.model.forward(as_tensor)
                label_index = torch.argmax(model_output).item()
                scores = {label: val.item() for label, val in zip(self.image_labels, model_output[0], strict=True)}
                label = self.image_labels[label_index]
                return label, scores

    @classmethod
    def create_dummy_model(cls):
        input_size = 128
        return CNNClassifier(
            num_classes=6,
            channels_in=3,
            channels_out=[32, 64, 128],
            conv_kernel_size=3,
            pool_size=2,
            linear_layer_neurons=256,
            input_image_size=input_size,
        )

    @classmethod
    def create_dummy_tranform(cls):
        input_size = 128
        return transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])

    @classmethod
    def create_dummy_labels(cls):
        return ["graphs_d", "graphs_h", "graphs_l", "graphs_s", "graphs_v", "graphs_val"]


class DoclingClassifier(ImageClassifier):
    """
    An image classifier that uses Docling's image classification model
    """
    def __init__(self):
        model_id = "docling-project/DocumentFigureClassifier-v2.5"
        self.model = EfficientNetForImageClassification.from_pretrained(model_id)
        self.model.eval()

        self.image_labels = self.model.config.id2label
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.47853944, 0.4732864, 0.47434163],
                ),
            ]
        )

    def classify_image(self, image_path: _typing.StrOrBytesPath) -> tuple[str|int, dict]:
        with Image.open(image_path) as im, torch.no_grad():
            im = im.convert("RGB")
            as_tensor = torch.unsqueeze(self.transform(im), 0)
            logits = self.model(as_tensor).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        label = self.image_labels[pred_id]
        scores_dict = {self.image_labels[i]: probs[0, i].item() for i in range(probs.shape[-1])}
        return label, scores_dict


if __name__ == "__main__":
    classifier = DoclingClassifier()
    im_path = "cnn_data_split/val/graphs_val/0a14bb795a27.jpg"
    print(classifier.classify_image(im_path))
