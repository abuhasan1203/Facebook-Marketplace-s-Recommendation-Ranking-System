from torchvision import transforms
from typing import Tuple, Union
import torch

class ImageProcessor:
    """
    ImageProcessor class is used to apply a series of image processing
    transformations to an image. Transforms can be added to the pipeline in any
    order and the pipeline can be applied to an image using the __call__ method.
    """
    def __init__(self):
        self.transform_pipeline = transforms.Compose([])

    def add_transform(
        self,
        name: str,
        args: Union[Tuple, dict]
    ) -> None:
        """
        Add a transform to the pipeline.

        :param name:
            name of the transform class.
        :param args:
            arguments to pass to the transform class constructor.
        """
        transform = None
        if name == "Resize":
            transform = transforms.Resize(args)
        elif name == "CenterCrop":
            transform = transforms.CenterCrop(args)
        elif name == "ToTensor":
            transform = transforms.ToTensor()
        elif name == "Normalize":
            transform = transforms.Normalize(args[0], args[1])
        elif name == "Lambda":
            transform = transforms.Lambda(self.repeat_channel)
        else:
            raise ValueError(f"Invalid transform name: {name}")
        self.transform_pipeline.transforms.append(transform)

    @staticmethod
    def repeat_channel(x):
        """
        Transformation for gray images.
        """
        return x.repeat(3, 1, 1)

    def __call__(
        self,
        img
    ) -> torch.Tensor:
        """
        Apply the transform pipeline to an image.

        :param img:
            image to be processed.
            
        :return:
            processed image.
        """
        img = self.transform_pipeline(img)
        return img[None, :, :, :]