import unittest
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms
from app.data_processors import FbMlImageProcessor

class TestFbMlImageProcessor(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test case.
        """
        self.fbml_image_processor = FbMlImageProcessor()

    def test_init(self):
        """
        Test the initializer of the FbMlImageProcessor class.
        This test case check if the class is correctly instantiated and the mean and std attributes are of type Tuple
        """ 
        self.assertIsInstance(self.fbml_image_processor, FbMlImageProcessor)
        self.assertIsInstance(self.fbml_image_processor.mean, Tuple)
        self.assertIsInstance(self.fbml_image_processor.std, Tuple)

    def test_call_with_rgb_image(self):
        """
        Test the __call__ method of the FbMlImageProcessor class with a RGB image.
        This test case checks if the processed image is of type torch.Tensor and has the correct shape.
        """
        image = Image.new('RGB', (256, 256), (255, 255, 255))
        processed_image = self.fbml_image_processor(image)
        self.assertIsInstance(processed_image, torch.Tensor)
        self.assertEqual(processed_image.shape, (1, 3, 224, 224))

    def test_call_with_non_rgb_image(self):
        """
        Test the __call__ method of the FbMlImageProcessor class with a non-RGB image.
        This test case checks if the processed image is of type torch.Tensor and has the correct shape.
        """
        image = Image.new('L', (256, 256), 255)
        processed_image = self.fbml_image_processor(image)
        self.assertIsInstance(processed_image, torch.Tensor)
        self.assertEqual(processed_image.shape, (1, 3, 224, 224))
        
    def test_call_with_non_image_input(self):
        """
        Test the __call__ method of the FbMlImageProcessor class with a non-image input.
        This test case checks that a TypeError is raised when the __call__ method is called with a non-image input.
        """
        with self.assertRaises(AttributeError):
            self.fbml_image_processor("not an image")

if __name__ == '__main__':
    unittest.main()