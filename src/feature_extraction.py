import numpy as np
import torch
import torchvision.models as models


class FeatureExtractor:
    def __init__(
            self,
            model_name: str = "resnet18"
    ):
        self.model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

    def extract_features(
            self,
            image: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            features = self.model(image.unsqueeze(0))
        return features.squeeze().numpy()
