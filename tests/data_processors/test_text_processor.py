import unittest
import torch
from modelling.data_processors.text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        """
        Create an instance of the TextProcessor class to be used in the test methods
        """
        self.tp = TextProcessor(model_name='bert-base-uncased', max_length=10)

    def test_text_processing(self):
        """
        Test that the TextProcessor class correctly processes a text input and returns a torch.Tensor object with the expected shape
        """
        text = "This is some text to be processed"
        result = self.tp(text)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.tp.model.config.hidden_size)
        self.assertEqual(result.shape[2], 10)

    def test_max_length(self):
        """
        Test that the TextProcessor class correctly truncates text inputs that are longer than the specified max_length of 10
        """
        text = "This is some text to be processed, it is longer than the max_length of 10"
        result = self.tp(text)
        self.assertEqual(result.shape[2], 10)

    def test_model_name(self):
        """
        Test that the TextProcessor class can use different pre-trained models by passing in the model_name argument
        """
        tp = TextProcessor(model_name='bert-base-cased', max_length=10)
        text = "This is some text to be processed"
        result = tp(text)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], tp.model.config.hidden_size)
        self.assertEqual(result.shape[2], 10)

if __name__ == '__main__':
    unittest.main()
