import pickle
import torch
import torch.nn as nn
from typing import Dict

class TextClassifier(nn.Module):
    def __init__(
        self,
        max_length: int,
        num_classes: int,
        input_size: int,
        decoder_file,
        fc = None,
        device: str = 'cpu' 
    ):
        """
        Initialise text classifier object.

        :param num_classes:
            Number of output classes.
        :param input_size:
            Input size of text.
        :param decoder_file:
            A pickle file containing a decoder for ouputting class.
        :param max_length:
            Max length for BERT transformation.
        :param fc:
            Fully connected layer.
        :param device:
            Default: cpu. Specify if needed.
        """
        super(TextClassifier, self).__init__()
        self.max_length = max_length
        self.num_classes = num_classes
        self.input_size = input_size
        self.decoder_file = decoder_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.device = torch.device(device)

        # Define the fully connected layer
        if not fc:
            if self.max_length == 100:
                self.fc = nn.Sequential(
                    nn.Linear(384, 128),
                    nn.ReLU(),
                    nn.Linear (128, self.num_classes)
                )
            elif self.max_length == 50:
                self.fc = nn.Sequential(
                    nn.Linear(192, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.num_classes)
                )
        else:
            self.fc = fc

        # Define the main CNN architecture
        self.main = nn.Sequential(
            nn.Conv1d(self.input_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            self.fc
        ).to(self.device)

    def forward(
        self,
        inp: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model

        :param inp:
            input tensor

        :return:
            output tensor
        """
        x = self.main(inp)
        return x

    def predict(
        self,
        inp: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform prediction on input tensor

        :param inp:
            input tensor

        :return:
            prediction tensor
        """
        with torch.no_grad():
            x = self.forward(inp)
            return x

    def predict_proba(
        self,
        inp: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform prediction with probability on input tensor

        :param inp:
            input tensor

        :return: 
            probability tensor
        """
        with torch.no_grad():
            x = self.forward(inp)
            return torch.softmax(x, dim=1)

    def predict_classes(
        self,
        inp: torch.Tensor
    ) -> str:
        """
        Perform prediction and return the class label

        :param inp:
            input tensor

        :return:
            class label
        """
        if self.decoder is None:
            raise ValueError("Decoder not provided")

        with torch.no_grad():
            x = self.forward(inp)
            class_index = torch.argmax(x, dim=1)
            return self.decoder[int(class_index)]