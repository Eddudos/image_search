import cv2
import torch
from torchvision import transforms as T


class ImagePreprocessor:
    """
    Class for image transitioning from CV2 to torch 
    """

    def __init__(
            self,
            image_size=(224, 224)
    ):
        self.image_size = image_size
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(image_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # As model was trained on ImageNet
                # it is better to use this values
            ]
        )

    def preprocess(
            self,
            image_path
    ):
        image = cv2.imread(image_path)
        image = self.transforms(image)
        return image


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    print(preprocessor.preprocess("/home/usersp/Downloads/ExtraCases/e8fc4e90-f6f4-460d-bedd-b82905ef6474.jpg").size())
