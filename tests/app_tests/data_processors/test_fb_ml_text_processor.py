import unittest
from app.data_processors import FbMlTextProcessor

class TestFbMlTextProcessor(unittest.TestCase):
    """
    Test the functionality of the FbMlTextProcessor class.
    """
    def setUp(self):
        """
        Create an instance of the FbMlTextProcessor class before each test.
        """
        self.processor = FbMlTextProcessor(max_length=50)

    def test_model_name(self):
        """
        Test that the model name is set correctly.
        """
        self.assertEqual(self.processor.model_name, 'bert-base-uncased')

    def test_max_length(self):
        """
        Test that the max_length is set correctly.
        """
        self.assertEqual(self.processor.max_length, 50)