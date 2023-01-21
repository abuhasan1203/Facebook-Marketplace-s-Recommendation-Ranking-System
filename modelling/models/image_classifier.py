import pickle
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict

class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        decoder_file,
        fc = None,
        device: str = 'cpu'
    ):
        """
        Initialise an Image Classifier model

        :param num_classes:
            Number of output classes
        :param decoder_file:
            A pickle file containing a dictionary for mapping class index to class label
        :param fc:
            A custom fully connected layer for the classifier
        :param device:
            The device to run the model on (default is "cpu")
        """
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes
        self.decoder_file = decoder_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.device = device
        self.image_model = models.resnet50(pretrained=True)
        in_features = self.image_model.fc.in_features
        if fc:
            self.image_model.fc = fc
        else:
            self.image_model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes)
            )
        self.main = nn.Sequential(self.image_model).to(self.device)

    def forward(
        self,
        image
    ) -> torch.Tensor:
        """
        Perform forward pass on the image

        :param image:
            input tensor

        :return:
            output tensor
        """
        x = self.main(image)
        return x

    def predict(
        self,
        image
    ) -> torch.Tensor:
        """
        Perform prediction on the input image

        :param image:
            input image (tensor)

        :return:
            prediction tensor
        """
        with torch.no_grad():
            x = self.forward(image)
            return x
    
    def predict_proba(
        self,
        image
    ) -> torch.Tensor:
        """
        Perform prediction on the input image and return class probabilities

        :param image:
            input image (tensor)

        :return: 
            probability tensor
        """
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(
        self,
        image
    ) -> str:
        """
        Perform prediction on the input image and return the predicted class label

        :param image:
            input image

        :return:
            class label
        """
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]