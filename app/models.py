from modelling.models import image_classifier, text_classifier, combined_classifier
import pickle
import torch

class FbMlImageClassifier(image_classifier.ImageClassifier):
    """
    A class for classifying images using a pre-trained model.
    """
    def __init__(
        self,
        decoder_file: str = 'modelling/decoders/image_decoder.pkl',
        state_dict_file: str = 'modelling/states/state_dict_image_model.pt'
    ):
        """
        Inherits from the ImageClassifier class.
    
        :param decoder_file::
            The file path to the decoder file.
        :param state_dict_file:
            The file path to the state dictionary file.
        """
        super().__init__(num_classes=13, decoder_file=decoder_file)
        self.decoder_file = decoder_file
        self.state_dict_file = state_dict_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = len(self.decoder)
        self.load_state_dict(torch.load(self.state_dict_file, map_location='cpu'), strict=False)

class FbMlTextClassifier(text_classifier.TextClassifier):
    """
    A class for classifying text.
    """
    def __init__(
        self,
        decoder_file: str = 'modelling/decoders/text_decoder.pkl',
        state_dict_file: str = 'modelling/states/state_dict_text_model.pt'
    ):
        """
        Inherits from the TextClassifier class.
    
        :param decoder_file::
            The file path to the decoder file.
        :param state_dict_file:
            The file path to the state dictionary file.
        """
        super().__init__(max_length=100, num_classes=13, input_size=768, decoder_file=decoder_file)
        self.decoder_file = decoder_file
        self.state_dict_file = state_dict_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = len(self.decoder)
        self.load_state_dict(torch.load(self.state_dict_file, map_location='cpu'), strict=False)
    
class FbMlCombinedClassifier(combined_classifier.ImageAndTextModel):
    """
    A class for classifying text and image.
    """
    def __init__(
        self,
        decoder_file: str = 'modelling/decoders/combined_decoder.pkl',
        state_dict_file: str = 'modelling/states/state_dict_combined_model.pt'
    ):
        """
        Inherits from the ImageAndTextModel class.
    
        :param decoder_file::
            The file path to the decoder file.
        :param state_dict_file:
            The file path to the state dictionary file.
        """
        super().__init__(max_length=50, num_classes=13, input_size=768, decoder_file=decoder_file)
        self.decoder_file = decoder_file
        self.state_dict_file = state_dict_file
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = len(self.decoder)
        self.load_state_dict(torch.load(self.state_dict_file, map_location='cpu'), strict=False)
