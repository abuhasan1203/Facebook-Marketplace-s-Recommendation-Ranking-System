import unittest
import pickle
from PIL import Image
from modelling.models.combined_classifier import ImageAndTextModel
from modelling.data_processors.image_processor import ImageProcessor
from modelling.data_processors.text_processor import TextProcessor

class TestImageAndTextModel(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test case with an instance of the ImageAndTextModel
        """
        self.ip = ImageProcessor()
        self.ip.add_transform("Resize", 256)
        self.ip.add_transform("CenterCrop", 224)
        self.ip.add_transform("ToTensor", None)
        self.ip.add_transform("Normalize", ((0.4217, 0.3923, 0.3633), (0.3117, 0.2967, 0.2931)))
        self.img = Image.open("tests/test_files/image.jpg")
        self.pimg = self.ip(self.img)

        self.tp = TextProcessor(model_name='bert-base-uncased', max_length=50)
        self.text = "This is some sample text to test."
        self.ptext = self.tp(self.text)

        self.decoder_file = 'tests/test_files/combined_decoder.pkl'
        with open(self.decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)
        self.num_classes = 13
        self.max_length = 50
        self.input_size = 768
        self.model = ImageAndTextModel(max_length=self.max_length, num_classes=self.num_classes, input_size=self.input_size, decoder_file=self.decoder_file)

    def test_forward(self):
        """
        Test the forward pass of the model
        """
        output = self.model(self.ptext, self.pimg)
        self.assertEqual(output.shape, (1, 13))

    def test_predict(self):
        """
        Test the predict method of the model
        """
        output = self.model.predict(self.ptext, self.pimg)
        self.assertEqual(output.shape, (1, 13))

    def test_predict_proba(self):
        """
        Test the predict_proba method of the model
        """
        output = self.model.predict_proba(self.ptext, self.pimg)
        self.assertEqual(output.shape, (1, 13))
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_predict_classes(self):
        """
        Test the predict_classes method of the model
        """
        output = self.model.predict_classes(self.ptext, self.pimg)
        self.assertIsInstance(output, str)

if __name__ == '__main__':
    unittest.main()
