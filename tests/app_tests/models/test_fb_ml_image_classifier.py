import unittest
import pickle
import torch
from app.models import FbMlImageClassifier

class TestFbMlImageClassifier(unittest.TestCase):

    def test_init(self):
        """
        Test the __init__ method of the FbMlImageClassifier class
        """
        fbml_classifier = FbMlImageClassifier()
        self.assertIsInstance(fbml_classifier, FbMlImageClassifier)

    def test_decoder_file(self):
        """
        Test the decoder file attribute of the FbMlImageClassifier class
        """
        decoder_file = 'modelling/decoders/image_decoder.pkl'
        fbml_classifier = FbMlImageClassifier(decoder_file=decoder_file)
        self.assertEqual(fbml_classifier.decoder_file, decoder_file)

    def test_state_dict_file(self):
        """
        Test the state_dict_file attribute of the FbMlImageClassifier class
        """
        state_dict_file = 'modelling/states/state_dict_image_model.pt'
        fbml_classifier = FbMlImageClassifier(state_dict_file=state_dict_file)
        self.assertEqual(fbml_classifier.state_dict_file, state_dict_file)

    def test_num_classes(self):
        """
        Test the num_classes attribute of the FbMlImageClassifier class
        """
        fbml_classifier = FbMlImageClassifier()
        with open(fbml_classifier.decoder_file, 'rb') as f:
            decoder = pickle.load(f)
        num_classes = len(decoder)
        self.assertEqual(fbml_classifier.num_classes, num_classes)

    def test_load_state_dict(self):
        """
        Test the load_state_dict method of the FbMlImageClassifier class
        """
        fbml_classifier = FbMlImageClassifier()
        state_dict = torch.load(fbml_classifier.state_dict_file, map_location='cpu')
        fbml_classifier.load_state_dict(state_dict, strict=False)
        self.assertIsNotNone(fbml_classifier.state_dict())

if __name__ == '__main__':
    unittest.main()

