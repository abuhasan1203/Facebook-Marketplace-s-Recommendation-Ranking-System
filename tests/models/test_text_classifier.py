import unittest
import pickle
import torch
from modelling.models.text_classifier import TextClassifier
from modelling.data_processors.text_processor import TextProcessor

class TestTextClassifier(unittest.TestCase):
    def setUp(self):
        """
        Create an instance of the TextClassifier class to be used in the test methods
        """
        self.tp = TextProcessor(model_name='bert-base-uncased', max_length=100)
        self.text = "This is some sample text to test."
        self.ptext = self.tp(self.text)

        self.decoder_file = 'tests/test_files/text_decoder.pkl'
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = 13
        self.input_size = 768
        self.max_length = 100
        self.tc = TextClassifier(max_length=self.max_length, num_classes=self.num_classes, input_size=self.input_size, decoder_file=self.decoder_file)

    def test_forward(self):
        """
        Test the forward method of the TextClassifier class
        """
        output_tensor = self.tc.forward(self.ptext)
        self.assertEqual(output_tensor.shape, (1, self.num_classes))

    def test_predict(self):
        """
        Test the predict method of the TextClassifier class
        """
        output_tensor = self.tc.predict(self.ptext)
        self.assertEqual(output_tensor.shape, (1, self.num_classes))

    def test_predict_proba(self):
        """
        Test the predict_proba method of the TextClassifier class
        """
        output_tensor = self.tc.predict_proba(self.ptext)
        self.assertEqual(output_tensor.shape, (1, self.num_classes))
        self.assertTrue(torch.all(torch.gt(output_tensor, 0)))
        self.assertTrue(torch.all(torch.lt(output_tensor, 1)))

    def test_predict_classes(self):
        """
        Test the predict_classes method of the TextClassifier class
        """
        output_str = self.tc.predict_classes(self.ptext)
        self.assertTrue(output_str in self.decoder.values())

if __name__ == '__main__':
    unittest.main()

