import cv2
import torch


class ImagePreprocessor:
    """
    Class for image transitioning from CV2 to torch 
    """
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image