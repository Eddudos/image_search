import fire
from image_preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from nearest_neighbors import NearestNeighborsSearch


class ImageSearchCLI:
    def __init__(self, data_path="data/test_data"):
        pass

    def search(self, image_path, class_name=None):
        pass

if __name__ == "__main__":
    fire.Fire(ImageSearchCLI)