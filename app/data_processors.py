from typing import Tuple
from modelling.data_processors import image_processor, text_processor

class FbMlImageProcessor(image_processor.ImageProcessor):
    """
    A class for processing images with standard pre-processing techniques.

    """
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.4217, 0.3923, 0.3633),
        std: Tuple[float, float, float] = (0.3117, 0.2967, 0.2931)
    ):
        """
        Inherits from the ImageProcessor class.
        
        :param mean:
            Mean values for each channel of the image.
        :param std:
            Standard deviation values for each channel of the image.
        """
        super().__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Applies the standard pre-processing techniques to the given image.
        
        :param img:
            The image to be processed
            
        :return:
            The processed image
        """
        if img.mode != 'RGB':
            self.add_transform('Resize', 256)
            self.add_transform('CenterCrop', 224)
            self.add_transform('ToTensor', None)
            self.add_transform('Lambda', None)
            self.add_transform('Normalize', (self.mean, self.std))
        else:
            self.add_transform('Resize', 256)
            self.add_transform('CenterCrop', 224)
            self.add_transform('ToTensor', None)
            self.add_transform('Normalize', (self.mean, self.std))
        img = self.transform_pipeline(img)
        return img[None, :, :, :]

class FbMlTextProcessor(text_processor.TextProcessor):
    """
    A class for processing text with standard pre-processing techniques.

    """
    def __init__(
        self,
        max_length
    ):
        """
        Inherits from the TextProcessor class.
        """
        super().__init__(model_name='bert-base-uncased', max_length=max_length)