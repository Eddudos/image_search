import os

import fire
import numpy as np

from feature_extraction import FeatureExtractor
from image_preprocessing import ImagePreprocessor
from nearest_neighbors import NearestNeighborsSearch


class ImageSearchCLI:
    """
    Класс для реализации CLI приложения поиска похожих изображений
    """

    def __init__(
            self,
            data_path: str = "data/test_data",
            model_name: str = "resnet18",
            embedding_dim: int = 512
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Создание экземпляров классов для предобработки
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor(self.model_name)
        self.nn_search = NearestNeighborsSearch(self.embedding_dim)

        # Дополнительный словарь для хранения соответствия индексов и путей к изображениям
        self.image_paths = {}
        self.load_data_and_create_index()

    def load_data_and_create_index(
            self
    ):
        index_counter = 0
        concatenated_embeddings = np.empty((0, 512))

        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)

                    # Сохраняем путь к изображению вместе с его индексом
                    self.image_paths[index_counter] = (image_path, class_name)
                    # Загрузка и предобработка изображения
                    image = self.preprocessor.preprocess(image_path)

                    # Извлечение признаков
                    embedding = self.feature_extractor.extract_features(image)
                    embedding = np.expand_dims(embedding, axis=0)

                    # Добавление эмбеддинга в индекс
                    concatenated_embeddings = np.concatenate((concatenated_embeddings, embedding), axis=0)
                    index_counter += 1
        self.nn_search.add_embeddings(concatenated_embeddings)

    def search(
            self,
            image_path: str,
            class_name: str = None,
            top_k: int = 6
    ) -> list[str]:
        image = self.preprocessor.preprocess(image_path)
        query_embedding = self.feature_extractor.extract_features(image)
        neighbors_indices = self.nn_search.search(query_embedding, top_k)

        print(f'Indices of nearest embeddings: {neighbors_indices} \n')

        found_image_paths = []
        for idx in neighbors_indices:
            path, image_class = self.image_paths[idx]
            if class_name is None or image_class == class_name:
                found_image_paths.append(path)
        return found_image_paths


if __name__ == "__main__":
    fire.Fire(ImageSearchCLI)
