import unittest
from PIL import Image
import torch
import torchvision.transforms as transforms
from modelling.data_processors.image_processor import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test case with an image, an ImageProcessor instance, and a path to the image file
        """
        self.img_path = "tests/test_files/image.jpg"
        self.img = Image.open(self.img_path)
        self.img_processor = ImageProcessor()

    def test_add_transform(self):
        """
        Test that the add_transform method is adding the correct transform to the pipeline
        """
        self.img_processor.add_transform("Resize", (256, 256))
        self.assertEqual(len(self.img_processor.transform_pipeline.transforms), 1)

        self.img_processor.add_transform("CenterCrop", (224, 224))
        self.assertEqual(len(self.img_processor.transform_pipeline.transforms), 2)

        self.img_processor.add_transform("ToTensor", None)
        self.assertEqual(len(self.img_processor.transform_pipeline.transforms), 3)

        self.img_processor.add_transform("Normalize", ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.assertEqual(len(self.img_processor.transform_pipeline.transforms), 4)
    
    def test_apply_transforms(self):
        """
        Test that the transforms are applied correctly by checking the output type and shape
        """
        self.img_processor.add_transform("Resize", (256, 256))
        self.img_processor.add_transform("CenterCrop", (224, 224))
        self.img_processor.add_transform("ToTensor", None)
        self.img_processor.add_transform("Normalize", ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        output = self.img_processor(self.img)
        self.assertEqual(type(output), torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 224, 224))
        
    def test_invalid_transform_name(self):
        """
        Test that the ValueError is raised when an invalid transform name is passed to the add_transform method.
        """
        with self.assertRaises(ValueError) as context:
            self.img_processor.add_transform("InvalidTransform", None)
        self.assertTrue("Invalid transform name" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
