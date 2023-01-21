import unittest
import pickle
import torch
from app.models import FbMlTextClassifier

class TestFbMlTextClassifier(unittest.TestCase):

    def setUp(self):
        """
        Initialize the test case.
        """
        self.decoder_file = 'modelling/decoders/text_decoder.pkl'
        self.state_dict_file = 'modelling/states/state_dict_text_model.pt'
        self.clf = FbMlTextClassifier(decoder_file=self.decoder_file, state_dict_file=self.state_dict_file)

    def test_decoder_file(self):
        """
        Test the decoder file attribute of the FbMlTextClassifier class
        """
        self.assertEqual(self.clf.decoder_file, self.decoder_file)

    def test_state_dict_file(self):
        """
        Test the state_dict_file attribute of the FbMlTextClassifier class
        """
        self.assertEqual(self.clf.state_dict_file, self.state_dict_file)

    def test_num_classes(self):
        """
        Test the num_classes attribute of the FbMlTextClassifier class
        """
        with open(self.decoder_file, 'rb') as f:
            decoder = pickle.load(f)
        num_classes = len(decoder)
        self.assertEqual(self.clf.num_classes, num_classes)

    def test_load_state_dict(self):
        """
        Test the load_state_dict method of the FbMlTextClassifier class
        """
        state_dict = torch.load(self.state_dict_file, map_location='cpu')
        self.clf.load_state_dict(state_dict, strict=False)
        self.assertIsNotNone(self.clf.state_dict())

if __name__ == '__main__':
    unittest.main()

