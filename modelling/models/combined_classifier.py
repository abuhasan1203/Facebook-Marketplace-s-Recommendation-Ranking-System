import pickle
import torch
import torch.nn as nn
from torchvision import models
from modelling.models.text_classifier import TextClassifier
from modelling.models.image_classifier import ImageClassifier

class ImageAndTextModel(nn.Module):
    def __init__(
        self,
        max_length: int,
        num_classes: int,
        input_size: int,
        decoder_file,
        device: str = 'cpu'
    ):
        """
        A model that combines image and text features for multi-modal classification

        :param num_classes:
            Number of classes in the final output
        :param input_size:
            Size of the input features
        :param decoder_file:
            A pickle file containing a dictionary to decode the final output
        :param device:
            The device to run the model on ('cpu' or 'cuda')
        """
        super(ImageAndTextModel, self).__init__()

        self.max_length = max_length
        self.num_classes = num_classes
        self.input_size = input_size
        self.decoder_file = decoder_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.device = device

        text_classifier_fc = nn.Sequential(
            nn.Linear(192, 128)
        )
        self.text_classifier = TextClassifier(
            max_length=self.max_length,
            num_classes=self.num_classes,
            input_size=self.input_size,
            decoder_file=self.decoder_file,
            fc=text_classifier_fc
        ).to(self.device)

        image_model = models.resnet50(pretrained=True)
        in_features = image_model.fc.in_features
        image_classifier_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.image_classifier = ImageClassifier(
            num_classes=self.num_classes,
            decoder_file=self.decoder_file,
            fc=image_classifier_fc
        ).to(self.device)

        self.main = nn.Sequential(
            nn.Linear(256, self.num_classes)
        ).to(self.device)

    def forward(
        self,
        text_inp,
        image_inp
    ) -> torch.Tensor:
        """
        Forward pass of the model

        :param text_inp:
            text input
        :param image_inp:
            image input

        :return:
            output tensor
        """
        text_features = self.text_classifier(text_inp)
        image_features = self.image_classifier(image_inp)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(
        self,
        text_inp,
        image_inp
    ) -> torch.Tensor:
        """
        Make a prediction

        :param text_inp:
            text input
        :param image_inp:
            image input

        :return:
            prediction tensor
        """
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return x

    def predict_proba(
        self,
        text_inp,
        image_inp
    ) -> torch.Tensor:
        """
        Make a prediction and return the probability distribution

        :param text_inp:
            text input
        :param image_inp:
            image input

        :return: 
            probability tensor
        """
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return torch.softmax(x, dim=1)

    def predict_classes(
        self,
        text_inp,
        image_inp
    ) -> str:
        """
        Make a prediction and return the class label

        :param text_inp:
            text input
        :param image_inp:
            image input

        :return:
            class label
        """
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return self.decoder[int(torch.argmax(x, dim=1))]