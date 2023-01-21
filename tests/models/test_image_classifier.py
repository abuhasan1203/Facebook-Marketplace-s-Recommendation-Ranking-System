import unittest
import pickle
from PIL import Image
import torch
from modelling.models.image_classifier import ImageClassifier
from modelling.data_processors.image_processor import ImageProcessor

class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        """
        Create an instance of the ImageClassifier class to be used in the test methods
        """
        self.ip = ImageProcessor()
        self.ip.add_transform("Resize", 256)
        self.ip.add_transform("CenterCrop", 224)
        self.ip.add_transform("ToTensor", None)
        self.ip.add_transform("Normalize", ((0.4217, 0.3923, 0.3633), (0.3117, 0.2967, 0.2931)))
        self.img = Image.open("tests/test_files/image.jpg")
        self.pimg = self.ip(self.img)

        self.decoder_file = 'tests/test_files/image_decoder.pkl'
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = 13
        self.image_classifier = ImageClassifier(num_classes=self.num_classes, decoder_file=self.decoder_file)

    def test_forward(self):
        """
        Test the forward method of the ImageClassifier class
        """
        output = self.image_classifier(self.pimg)
        self.assertEqual(output.shape, (1, self.num_classes))

    def test_predict(self):
        """
        Test the predict method of the ImageClassifier class
        """
        output = self.image_classifier.predict(self.pimg)
        self.assertEqual(output.shape, (1, self.num_classes))

    def test_predict_proba(self):
        """
        Test the predict_proba method of the ImageClassifier class
        """
        output = self.image_classifier.predict_proba(self.pimg)
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())
        self.assertTrue(torch.isclose(torch.sum(output, dim=1), torch.ones(1)).all())

    def test_predict_classes(self):
        """
        Test the predict_classes method of the ImageClassifier class
        """
        output = self.image_classifier.predict_classes(self.pimg)
        self.assertIsInstance(output, str)
        self.assertIn(output, self.decoder.values())

if __name__ == '__main__':
    unittest.main()